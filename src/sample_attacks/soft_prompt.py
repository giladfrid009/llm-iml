from src.adv_model import AdvModel
from src.sample_attacks.sample_attack import SampleAttack, SampleOutput

import inspect
from typing import Callable, Iterable
from tqdm.auto import tqdm
import torch
import copy


from transformers.tokenization_utils_base import BatchEncoding
from transformers.cache_utils import DynamicCache


class SoftPrompt(SampleAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        steps: int = 100,
        early_stopping: bool = False,
        mixed_precision: bool = False,
        kv_caching: bool = True,
        verbose: bool = True,
    ):
        super().__init__(adv_model, verbose)

        self.steps = steps
        self.optim_factory = optim_factory
        self.early_stopping = early_stopping
        self.mixed_precision = mixed_precision
        self.kv_caching = kv_caching

    def get_hparams(self) -> dict:
        dummy_optim = self.optim_factory([torch.zeros(1)])
        return {
            "name": self.__class__.__name__,
            "steps": self.steps,
            "early_stopping": self.early_stopping,
            "mixed_precision": self.mixed_precision,
            "kv_caching": self.kv_caching,
            "optim": dummy_optim.state_dict()["param_groups"][0],
            "optim/name": dummy_optim.__class__.__name__,
            "optim_factory": inspect.getsource(self.optim_factory),
        }

    def _initialize_embeddings(
        self,
        num_inputs: int,
        init_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if init_embeds is not None:
            embeddings = init_embeds.clone().detach()
            embeddings.requires_grad_(True)
            return embeddings

        return torch.randn(
            size=(num_inputs, self.adv_model.num_tokens, self.adv_model.adv_embedder.embed_dim),
            device=self.adv_model.device,
            dtype=self.adv_model.adv_embedder.embed_dtype,
            requires_grad=True,
        )

    @torch.no_grad()
    def _compute_cache(self, encodings: BatchEncoding) -> BatchEncoding:
        """
        Compute the kv-cache for the constant part of the input.

        Args:
            encodings (BatchEncoding): encodings containing input tokens and masks,
                as returned from `self.adv_model.tokenize()`.

        Returns:
            BatchEncoding: encodings containing the kv-cache for the constant part of the input,
                as well as the remaining input tokens and masks. The remaining input tokens
                and masks are only the variable (non-cached) part of the input.
        """

        kv_idx = encodings.const_idx.min().item()

        kv_result = self.adv_model.forward(
            encodings.input_ids[:, :kv_idx],
            encodings.attention_mask[:, :kv_idx],
            adv_mask=None,
            adv_embeds=None,
            use_cache=True,
        )

        new_data = {
            "input_ids": encodings.input_ids[:, kv_idx:],
            "attention_mask": encodings.attention_mask,  # we need the full attention mask
            "adv_mask": encodings.adv_mask[:, kv_idx:],
            "kv_cache": kv_result.past_key_values,
        }

        if "target_mask" in encodings:
            new_data["target_mask"] = encodings.target_mask[:, kv_idx:]

        return BatchEncoding(new_data)

    def _masked_select(
        self,
        encodings: BatchEncoding,
        mask: torch.Tensor,
    ) -> BatchEncoding:
        """
        Selects elements from the encodings based on the provided mask.
        Mask is applied to the batch dimension of all tensor elements in the encodings.
        """
        new_data = {}
        for key, value in encodings.data.items():
            if isinstance(value, torch.Tensor):
                new_data[key] = value[mask]

            elif key == "kv_cache" and value is None:
                new_data[key] = None

            elif key == "kv_cache" and isinstance(value, DynamicCache):
                indices = mask.nonzero(as_tuple=True)[0]
                cache = value.batch_select_indices(indices)
                new_data[key] = cache

            else:
                raise TypeError(f"Unsupported type {type(value)} for key {key} in encodings")

        return BatchEncoding(new_data)

    @torch.no_grad()
    def _check_early_stopping(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check which samples have achieved perfect target matching.

        Args:
            logits: Predicted logits
            target_ids: Target token IDs
            target_mask: Target mask

        Returns:
            Boolean tensor indicating which samples are finished
        """
        match_mask = logits.argmax(dim=-1) == target_ids
        match_mask.masked_fill_(~target_mask, True)  # Ignore non-target tokens
        return match_mask.all(dim=-1).flatten()

    def criterion(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        CE loss for the logits and target_ids, masked by target_mask.
        """
        # extract only targets
        logits = logits[target_mask].reshape(-1, logits.size(-1))
        target_ids = target_ids[target_mask].flatten()

        # compute token-wise loss
        flat_losses = torch.nn.functional.cross_entropy(logits, target_ids, reduction="none")

        # scatter losses back to the original shape and compute sample-mean
        loss_matrix = torch.zeros_like(target_mask, dtype=logits.dtype)
        loss_matrix[target_mask] = flat_losses
        loss = torch.sum(loss_matrix.sum(dim=-1) / target_mask.sum(dim=-1))
        return loss

    def fit(
        self,
        conversations: list[list[dict[str, str]]],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> SampleOutput:
        # initialize optimized embeddings
        adv_embeds = self._initialize_embeddings(
            num_inputs=len(conversations),
            init_embeds=init_embeds,
        )

        # create optimizer and scaler
        scaler = torch.GradScaler(enabled=self.mixed_precision)
        optim = self.optim_factory([adv_embeds])

        # tokenize
        conversations = self.adv_model.inject_tokens(conversations)
        encodings = self.adv_model.tokenize(conversations, target_texts)

        # compute kv-cache if enabled
        if self.kv_caching:
            with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                encodings = self._compute_cache(encodings)
        # early stopping state
        finished = torch.zeros(len(conversations), dtype=torch.bool, device=self.device)
        optim_embeds = adv_embeds

        with tqdm(range(self.steps), disable=not self.verbose, leave=False, desc="Attack") as pbar:
            for step in pbar:
                optim.zero_grad()

                # NOTE: need to copy kv-cache since forward modifies it in-place
                step_encodings = encodings.copy()
                step_encodings["kv_cache"] = copy.deepcopy(step_encodings.get("kv_cache", None))

                # select only unfinished samples if early stopping is enabled
                if self.early_stopping:
                    step_encodings = self._masked_select(step_encodings, ~finished)
                    optim_embeds = adv_embeds[~finished]

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    # forward pass
                    result = self.adv_model.forward(
                        input_ids=step_encodings.input_ids,
                        attention_mask=step_encodings.attention_mask,
                        adv_mask=step_encodings.adv_mask,
                        past_key_values=step_encodings.kv_cache,
                        adv_embeds=optim_embeds,
                    )

                    # align predicted logits and target_ids
                    logits: torch.Tensor = result.logits[:, :-1]  # remove new token
                    target_ids = step_encodings.input_ids[:, 1:]  # remove BOS token
                    target_mask = step_encodings.target_mask[:, 1:]  # remove BOS token

                    # update early stopping based on predictions
                    if self.early_stopping:
                        finished_status = self._check_early_stopping(logits, target_ids, target_mask)
                        finished[~finished] = finished_status
                        if finished.all():
                            # break early
                            pbar.n = pbar.total
                            pbar.close()
                            break

                    loss = self.criterion(
                        logits=logits,
                        target_ids=target_ids,
                        target_mask=target_mask,
                    )

                # backward pass and optimization step
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                pbar.set_postfix({"loss": loss.item()})

        return SampleOutput(conversations, adv_embeds.detach())
