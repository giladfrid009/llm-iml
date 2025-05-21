from typing import Callable, Iterable
from src.adver_model import AdverModel
from src.attack import Attack
from tqdm.auto import tqdm
import torch
import copy


class OptimAttack(Attack):
    def __init__(
        self,
        adv_model: AdverModel,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        steps: int = 100,
        mixed_precision: bool = True,
        caching: bool = True,
        silent: bool = False,
    ):
        super().__init__(adv_model, silent)

        self.steps = steps
        self.optim_factory = optim_factory
        self.mixed_precision = mixed_precision
        self.caching = caching # TODO: implement switch to turn on or off kv-caching

    def fit(
        self,
        input_texts: list[str],
        target_texts: list[str],
        adv_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:

        token_dict = self.adv_model.tokenize(input_texts, target_texts)

        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, enabled=self.mixed_precision),
        ):
            # construct kv-cache for the constant, first part of the batch
            kv_idx = token_dict["const_idx"].min().item()
            kv_result = self.adv_model.forward(
                token_dict["input_ids"][:, :kv_idx],
                token_dict["attention_mask"][:, :kv_idx],
                use_cache=True,
            )
            kv_cache = copy.deepcopy(kv_result.past_key_values)

        if adv_embeds is not None:
            adv_embeds = adv_embeds.clone().detach()
            adv_embeds.requires_grad_(True)
        else:
            adv_embeds = self.init_embedding(num_inputs=len(input_texts))

        scaler = torch.GradScaler(enabled=self.mixed_precision)
        optim = self.optim_factory([adv_embeds])

        with tqdm(range(self.steps), disable=self.silent, leave=False) as pbar:
            for step in pbar:

                optim.zero_grad()

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):

                    result = self.adv_model.forward(
                        input_ids=token_dict["input_ids"][:, kv_idx:],
                        attention_mask=token_dict["attention_mask"],
                        past_key_values=copy.deepcopy(kv_cache),  # NOTE: important to copy as forwrd modifies the cache in-place
                        adv_embeds=adv_embeds,
                        adv_mask=token_dict["adv_mask"][:, kv_idx:],
                    )

                    pred_logits, target_ids = self.align_preds(
                        result.logits,
                        token_dict["input_ids"][:, kv_idx:],
                        token_dict["target_mask"][:, kv_idx:],
                    )

                    loss = torch.nn.functional.cross_entropy(pred_logits, target_ids)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                pbar.set_postfix({"loss": loss.item()})

        return adv_embeds.detach()
