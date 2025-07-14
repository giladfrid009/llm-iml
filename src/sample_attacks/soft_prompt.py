from src.adv_model import AdvModel
from src.sample_attacks.sample_attack import SampleAttack, SampleOutput

from typing import Callable, Iterable
from tqdm.auto import tqdm
import torch
import copy

from transformers.cache_utils import Cache, DynamicCache

LegacyCache = tuple[tuple[torch.Tensor], tuple[torch.Tensor]]


class SoftPrompt(SampleAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        steps: int = 100,
        early_stopping: bool = True,
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

    def _init_embedding(self, num_inputs: int) -> torch.Tensor:
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

    def criterion(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        # align prdicted logits and target_ids
        logits = logits[:, :-1]  # remove new token
        target_ids = input_ids[:, 1:]  # remove BOS token
        target_mask = target_mask[:, 1:]  # remove BOS token

        # compute CE loss
        loss_matrix = torch.nn.functional.cross_entropy(logits.swapdims(-1, -2), target_ids, reduction="none")
        loss_matrix = loss_matrix * target_mask.bool()
        loss = torch.mean(loss_matrix.sum(dim=-1) / target_mask.sum(dim=-1))
        return loss

    def fit(
        self,
        conversations: list[list[dict[str, str]]],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> SampleOutput:
        # TODO: IMPORTANT: add early stopping. If logits.argmax() == target_ids, then stop optimizing for this sample
        # whats cool is that it doesnt require us to call expensive generate()
        # to filter the cache object for early stopping see:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
        # in the _contrastive_search method, they use DynamicCache.batch_select_indices()
        # Note that early stopping can be loss-based instead. That will not require additional forward passes for the
        # early stopping check.

        token_dict = self.adv_model.tokenize(conversations, target_texts)

        if self.kv_caching:
            with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                token_dict = self._compute_cache(token_dict)

        adv_embeds = init_embeds
        if adv_embeds is not None:
            adv_embeds = adv_embeds.clone().detach()
            adv_embeds.requires_grad_(True)
        else:
            adv_embeds = self._init_embedding(num_inputs=len(conversations))
            adv_embeds.requires_grad_(True)

        scaler = torch.GradScaler(enabled=self.mixed_precision)
        optim = self.optim_factory([adv_embeds])

        with tqdm(range(self.steps), disable=not self.verbose, leave=False, desc="Attack") as pbar:
            for step in pbar:
                optim.zero_grad()

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    past_keys_values: Cache | LegacyCache | None = None
                    if self.kv_caching:
                        # NOTE: need to copy since forward modifies it in-place
                        past_keys_values = copy.deepcopy(token_dict["kv_cache"])

                    result = self.adv_model.forward(
                        input_ids=token_dict["input_ids"],
                        attention_mask=token_dict["attention_mask"],
                        past_key_values=past_keys_values,
                        adv_mask=token_dict["adv_mask"],
                        adv_embeds=adv_embeds,
                    )

                    loss = self.criterion(
                        logits=result.logits,
                        input_ids=token_dict["input_ids"],
                        target_mask=token_dict["target_mask"],
                    )

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                pbar.set_postfix({"loss": loss.item()})

        return SampleOutput(conversations=conversations, adv_embeds=adv_embeds.detach())
