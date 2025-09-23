from src.adv_model import AdvModel
from src.eval.evaluator import Evaluator
from src.config import GenConfig
from src.univ_attacks.univ_attack import UnivAttack
from src.fgsm_optim import FGSM

from typing import Any
import torch


class UnivSoftPrompt(UnivAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        optimizer: FGSM,
        evaluators: list[Evaluator],
        judge_metric: str | None = None,
        eval_freq: int | float = 1,
        mixed_precision: bool = True,
        gen_config: GenConfig | None = None,
        log_dir: str | None = None,
    ):
        super().__init__(
            adv_model=adv_model,
            evaluators=evaluators,
            judge_metric=judge_metric,
            eval_freq=eval_freq,
            mixed_precision=mixed_precision,
            gen_config=gen_config,
            log_dir=log_dir,
        )

        self.optimizer = optimizer
        
        self.logger.register_hparams({"univ_soft_prompt/optimizer": self.optimizer.__class__.__name__})
        self.logger.register_hparams({"optim/name": self.optimizer.__class__.__name__})
        self.logger.register_hparams({f"optim/{k}": v for k, v in self.optimizer.param_groups[0].items()})

    def criterion(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        # align predicted logits and target_ids
        logits = logits[:, :-1]  # remove new token
        target_ids = input_ids[:, 1:]  # remove BOS token
        target_mask = target_mask[:, 1:]  # remove BOS token

        # compute flat CE loss
        target_logits = logits[target_mask].view(-1, logits.size(-1))
        target_ids = target_ids[target_mask].view(-1)
        loss = torch.nn.functional.cross_entropy(target_logits, target_ids, reduction="mean")
        return loss

    def optim_step(self, data: dict[str, list[Any]], epoch_num: int, batch_num: int) -> float | None:
        self.optimizer.zero_grad()

        # construct input conversations
        input_text, target_text = data["prompt"], data["target"]
        conversations = [[{"role": "user", "content": prm}] for prm in input_text]
        token_dict = self.adv_model.tokenize(conversations, target_text)

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            result = self.adv_model.forward(
                input_ids=token_dict["input_ids"],
                attention_mask=token_dict["attention_mask"],
                adv_mask=token_dict["adv_mask"],
            )

            loss = self.criterion(
                logits=result.logits,
                input_ids=token_dict["input_ids"],
                target_mask=token_dict["target_mask"],
            )

        # grad step
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return loss.item()
