from src.adv_model import AdvModel
from src.sample_attacks.sample_attack import SampleAttack, SampleOutput

from typeguard import check_type
from typing import Callable, Iterable, Any
from tqdm.auto import tqdm
import torch
import copy

from transformers.cache_utils import DynamicCache

LegacyCache = tuple[tuple[torch.Tensor], tuple[torch.Tensor]]


class SoftPrompt(SampleAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        steps: int = 100,
        early_stopping: bool = False,
        mixed_precision: bool = True,
        kv_caching: bool = True,
        verbose: bool = True,
    ):
        super().__init__(adv_model, verbose)

        self.steps = steps
        self.optim_factory = optim_factory
        self.early_stopping = early_stopping
        self.mixed_precision = mixed_precision
        self.kv_caching = kv_caching

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
    def _compute_cache(self, token_dict: dict[str, torch.Tensor]) -> dict:
        """
        Compute the kv-cache for the constant part of the input.

        Args:
            token_dict (dict[str, torch.Tensor]): Dictionary containing input tokens and masks,
                as returned from `self.adv_model.tokenize()`.

        Returns:
            dict: Dictionary containing the kv-cache for the constant part of the input,
                as well as the remaining input tokens and masks. The remaining input tokens
                and masks are only the variable (non-cached) part of the input.
        """

        kv_idx = token_dict["const_idx"].min().item()

        kv_result = self.adv_model.forward(
            token_dict["input_ids"][:, :kv_idx],
            token_dict["attention_mask"][:, :kv_idx],
            use_cache=True,
        )

        new_token_dict = {
            "input_ids": token_dict["input_ids"][:, kv_idx:],
            "attention_mask": token_dict["attention_mask"],  # we need the full attention mask
            "adv_mask": token_dict["adv_mask"][:, kv_idx:],
            "kv_cache": kv_result.past_key_values,
        }

        if "target_mask" in token_dict:
            new_token_dict["target_mask"] = token_dict["target_mask"][:, kv_idx:]

        return new_token_dict

    def _masked_select(
        self,
        token_dict: dict[str, Any],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | Any]:
        """
        Selects elements from the token_dict based on the provided mask.
        Mask is applied to the batch dimension of all elements in the token_dict.
        """
        new_dict = {}
        for key, value in token_dict.items():
            if isinstance(value, torch.Tensor):
                new_dict[key] = value[mask]

            elif isinstance(value, DynamicCache):
                indices = mask.nonzero(as_tuple=True)[0]
                cache = value.batch_select_indices(indices)
                new_dict[key] = cache

            elif check_type(value, LegacyCache):
                cache = (tuple(tensor[mask] for tensor in value[0]), tuple(tensor[mask] for tensor in value[1]))
                new_dict[key] = cache

            else:
                raise TypeError(f"Unsupported type {type(value)} for key {key} in token_dict")

        return new_dict

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
        sample_mean: bool = True,
    ) -> torch.Tensor:
        """
        CE loss for the logits and target_ids, masked by target_mask.
        """
        # extract only targets
        logits = logits[target_mask].reshape(-1, logits.size(-1))
        target_ids = target_ids[target_mask].flatten()

        # compute token-wise loss
        flat_losses = torch.nn.functional.cross_entropy(logits, target_ids, reduction="none")

        if not sample_mean:
            return flat_losses.mean()

        # scatter losses back to the original shape and compute sample-mean
        loss_matrix = torch.zeros_like(target_mask, dtype=logits.dtype)
        loss_matrix[target_mask] = flat_losses
        loss = torch.mean(loss_matrix.sum(dim=-1) / target_mask.sum(dim=-1))
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
        token_dict = self.adv_model.tokenize(conversations, target_texts)

        # compute kv-cache if enabled
        if self.kv_caching:
            with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                token_dict = self._compute_cache(token_dict)

        # early stopping state
        finished = torch.zeros(len(conversations), dtype=torch.bool, device=self.device)
        optim_embeds = adv_embeds

        with tqdm(range(self.steps), disable=not self.verbose, leave=False, desc="Attack") as pbar:
            for step in pbar:
                optim.zero_grad()

                # NOTE: need to copy kv-cache since forward modifies it in-place
                iter_data = token_dict.copy()
                iter_data["kv_cache"] = copy.deepcopy(iter_data.get("kv_cache", None))

                # select only unfinished samples if early stopping is enabled
                if self.early_stopping:
                    iter_data = self._masked_select(iter_data, ~finished)
                    optim_embeds = adv_embeds[~finished]

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    # forward pass
                    result = self.adv_model.forward(
                        input_ids=iter_data["input_ids"],
                        attention_mask=iter_data["attention_mask"],
                        adv_mask=iter_data["adv_mask"],
                        past_key_values=iter_data["kv_cache"],
                        adv_embeds=optim_embeds,
                    )

                    # align predicted logits and target_ids
                    logits: torch.Tensor = result.logits[:, :-1]  # remove new token
                    target_ids = iter_data["input_ids"][:, 1:]  # remove BOS token
                    target_mask = iter_data["target_mask"][:, 1:]  # remove BOS token

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
                        sample_mean=False,
                    )

                # backward pass and optimization step
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                pbar.set_postfix({"loss": loss.item()})

        return SampleOutput(conversations=conversations, adv_embeds=adv_embeds.detach())
