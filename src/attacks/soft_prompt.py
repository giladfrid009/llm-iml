from typing import Callable, Iterable
from tqdm.auto import tqdm
import torch
import copy

from transformers.tokenization_utils_base import BatchEncoding
from transformers.cache_utils import Cache, DynamicCache

LegacyCache = tuple[tuple[torch.Tensor], tuple[torch.Tensor]]

from src.adver_model import AdverModel
from src.attacks.attack import Attack


class SoftPrompt(Attack):
    def __init__(
        self,
        adv_model: AdverModel,
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

    def _align_preds(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aligns the predictions with the target tokens for loss calculation.

        Args:
            logits (torch.Tensor): Model output logits.
            input_ids (torch.Tensor): Input token IDs.
            target_mask (torch.Tensor): Mask indicating target tokens.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing aligned logits and target IDs.
        """
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        target_mask = target_mask[:, 1:]

        target_ids = input_ids[target_mask].view(-1)
        pred_logits = logits[target_mask].view(-1, logits.size(-1))
        return pred_logits, target_ids

    def _init_embedding(self, num_inputs: int, **kwargs) -> torch.Tensor:
        return torch.randn(
            size=(num_inputs, self.num_tokens, self.embed_dim),
            device=self.device,
            dtype=self.embed_dtype,
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

    def fit(
        self,
        conversations: list[list[dict[str, str]]],
        target_texts: list[str],
        embeds_init: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # TODO: important: add early stopping. If logits.argmax() == target_ids, then stop optimizing for this sample
        # whats cool is that it doesnt require us to call expensive generate()
        # to filter the cache object for early stopping see:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
        # in the _contrastive_search method, they use DynamicCache.batch_select_indices()

        token_dict = self.adv_model.tokenize(conversations, target_texts)

        if self.kv_caching:
            with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                token_dict = self._compute_cache(token_dict)

        adv_embeds = embeds_init
        if adv_embeds is not None:
            adv_embeds = adv_embeds.clone().detach()
            adv_embeds.requires_grad_(True)
        else:
            adv_embeds = self._init_embedding(num_inputs=len(conversations))
            adv_embeds.requires_grad_(True)

        self.adv_model.set_embeddings(adv_embeds)

        scaler = torch.GradScaler(enabled=self.mixed_precision)
        optim = self.optim_factory(self.adv_model.parameters())

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
                    )

                    pred_logits, target_ids = self._align_preds(
                        logits=result.logits,
                        input_ids=token_dict["input_ids"],
                        target_mask=token_dict["target_mask"],
                    )

                    # TODO: currently in the loss caclulation, we give uniform weights to all tokens.
                    # i think instead we should first average per-sequence and then average over the sequences.
                    # but it is an annoying implementation since we do mask_select here to choose the target tokens.
                    loss = torch.nn.functional.cross_entropy(pred_logits, target_ids)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                pbar.set_postfix({"loss": loss.item()})

        return adv_embeds.detach()
