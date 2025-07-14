from src.adv_model import AdvModel
from src.eval.evaluator import Evaluator
from src.config import GenConfig
from src.univ_attacks.univ_attack import UnivAttack

from typing import Any
import torch
from typing import Iterable, Callable


class UnivSoftPrompt(UnivAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        evaluators: list[Evaluator],
        eval_freq: int | float = 1,
        mixed_precision: bool = True,
        gen_config: GenConfig | None = None,
        log_dir: str | None = None,
    ):
        super().__init__(
            adv_model=adv_model,
            evaluators=evaluators,
            eval_freq=eval_freq,
            mixed_precision=mixed_precision,
            gen_config=gen_config,
            log_dir=log_dir,
        )

        self.optimizer = optim_factory([self.univ_embeds])
        self.logger.register_hparams({"iml/optimizer": self.optimizer.__class__.__name__})
        self.logger.register_hparams({"optim/name": self.optimizer.__class__.__name__})
        self.logger.register_hparams({f"optim/{k}": v for k, v in self.optimizer.param_groups[0].items()})

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

    def optim_step(self, data: dict[str, list[Any]], epoch_num: int, batch_num: int) -> float | None:
        self.optimizer.zero_grad()

        # construct input conversations
        input_text, target_text = data["prompt"], data["target"]
        input_convs = [[{"role": "user", "content": prm}] for prm in input_text]
        token_dict = self.adv_model.tokenize(input_convs, target_text)

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            result = self.adv_model.forward(
                input_ids=token_dict["input_ids"],
                attention_mask=token_dict["attention_mask"],
                adv_mask=token_dict["adv_mask"],
                return_dict=False,
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
