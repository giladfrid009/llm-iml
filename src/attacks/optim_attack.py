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
        kv_caching: bool = True,
        silent: bool = False,
    ):
        super().__init__(adv_model, silent)

        self.steps = steps
        self.optim_factory = optim_factory
        self.mixed_precision = mixed_precision
        self.kv_caching = kv_caching  # TODO: implement switch to turn on or off kv-caching

    def fit(
        self,
        input_texts: list[str],
        target_texts: list[str],
        embeds_init: torch.Tensor | None = None,
    ) -> torch.Tensor:

        token_dict = self.adv_model.tokenize(input_texts, target_texts)

        if self.kv_caching:
            with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                token_dict = self.compute_cache(token_dict)

        adv_embeds = embeds_init
        if adv_embeds is not None:
            adv_embeds = adv_embeds.clone().detach()
            adv_embeds.requires_grad_(True)
        else:
            adv_embeds = self.init_embedding(num_inputs=len(input_texts))
            adv_embeds.requires_grad_(True)

        scaler = torch.GradScaler(enabled=self.mixed_precision)
        optim = self.optim_factory([adv_embeds])

        with tqdm(range(self.steps), disable=self.silent, leave=False) as pbar:
            for step in pbar:

                optim.zero_grad()

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):

                    past_keys_values = None
                    if self.kv_caching:
                        # NOTE: need to copy since forward modifies in in-place
                        past_keys_values = copy.deepcopy(token_dict["kv_cache"])

                    result = self.adv_model.forward(
                        input_ids=token_dict["input_ids"],
                        attention_mask=token_dict["attention_mask"],
                        past_key_values=past_keys_values,
                        adv_embeds=adv_embeds,
                        adv_mask=token_dict["adv_mask"],
                    )

                    pred_logits, target_ids = self.align_preds(
                        logits=result.logits,
                        input_ids=token_dict["input_ids"],
                        target_mask=token_dict["target_mask"],
                    )

                    loss = torch.nn.functional.cross_entropy(pred_logits, target_ids)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                pbar.set_postfix({"loss": loss.item()})

        return adv_embeds.detach()
