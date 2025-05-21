from typing import Callable, Iterable
from src.embed_injector import EmbedInjector
from src.attack import Attack
from tqdm.auto import tqdm
import torch


class OptimAttack(Attack):
    def __init__(
        self,
        embed_injector: EmbedInjector,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        steps: int = 100,
        mixed_precision: bool = True,
        silent: bool = False,
    ):
        super().__init__(embed_injector, silent)

        self.steps = steps
        self.optim_factory = optim_factory
        self.mixed_precision = mixed_precision

    def init_embedding(self, orig_embeds: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            size=(orig_embeds.size(0), self.num_tokens, orig_embeds.size(-1)),
            device=self.device,
            dtype=orig_embeds.dtype,
            requires_grad=True,
        )

    def fit(
        self,
        input_texts: list[str],
        target_texts: list[str],
        init_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # TODO: KV-Cache doesnt work with the inputs_embeds for some reason
        # figure out how to make it work, or is it a lost cause?
        # if kv-cache doesnt work, then we should try to find a more packed way to store the
        # input embeddings

        embed_dict = self.embed_injector.embed_text(input_texts, target_texts)
        orig_embeds = embed_dict["inputs_embeds"]

        with torch.no_grad():
            kv_idx = embed_dict["const_idx"].min()
            kv_result = self.embed_injector.forward(
                orig_embeds[:, :kv_idx],
                embed_dict["attention_mask"][:, :kv_idx],
                use_cache=True,
            )
            kv_cache = kv_result.past_key_values

        if init_embedding is not None:
            adv_embed = init_embedding.clone().detach()
            adv_embed.requires_grad_(True)
        else:
            adv_embed = self.init_embedding(orig_embeds)

        scaler = torch.GradScaler(enabled=self.mixed_precision)
        optim = self.optim_factory([adv_embed])

        with tqdm(range(self.steps), disable=self.silent, leave=False) as pbar:

            for step in pbar:

                optim.zero_grad()

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):

                    inj_embeds = self.embed_injector.inject_embedding(
                        inp_embeds=orig_embeds,
                        adv_embeds=adv_embed,
                        adv_mask=embed_dict["adv_mask"],
                    )

                    result = self.embed_injector.forward(
                        # input_ids=embed_dict["input_ids"][:, :kv_idx],
                        inputs_embeds=inj_embeds[:, :kv_idx],
                        labels=embed_dict["input_ids"],
                        attention_mask=embed_dict["attention_mask"],
                        past_key_values=kv_cache,
                    )

                    pred_logits, target_ids = self.align_preds(result.logits, embed_dict["input_ids"], embed_dict["target_mask"])
                    loss = torch.nn.functional.cross_entropy(pred_logits, target_ids)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                pbar.set_postfix({"loss": loss.item()})

        return adv_embed.detach()
