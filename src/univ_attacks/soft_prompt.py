from src.adv_model import AdvModel
from src.eval.evaluator import Evaluator
from src.config import GenConfig
from src.univ_attacks.univ_attack import UnivAttack
from src.fgsm_optim import FGSM
from src.metric_logger import MetricLogger

from typing import Any
import torch


class UnivSoftPrompt(UnivAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        optimizer: FGSM,
        evaluators: list[Evaluator],
        eval_metric: str | None = None,
        eval_freq: int | float = 1,
        mixed_precision: bool = False,
        gen_config: GenConfig | None = None,
        metric_logger: MetricLogger | None = None,
    ):
        super().__init__(
            adv_model=adv_model,
            evaluators=evaluators,
            eval_metric=eval_metric,
            eval_freq=eval_freq,
            mixed_precision=mixed_precision,
            gen_config=gen_config,
            metric_logger=metric_logger,
        )

        self.optimizer = optimizer

        self.metric_logger.log_hparams(
            "univ_soft_prompt",
            optimizer=self.optimizer.__class__.__name__,
        )
        self.metric_logger.log_hparams(
            "optim",
            optimizer.state_dict()["param_groups"][0],
            name=self.optimizer.__class__.__name__,
        )

    def criterion(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        target_mask: torch.Tensor,
        sample_mean: bool = True,
    ) -> torch.Tensor:
        # align predicted logits and target_ids
        logits = logits[:, :-1]  # remove new token
        input_ids = input_ids[:, 1:]  # remove BOS token
        target_mask = target_mask[:, 1:]  # remove BOS token

        # extract only targets
        target_logits = logits[target_mask].view(-1, logits.size(-1))
        target_ids = input_ids[target_mask].view(-1)

        # compute token-wise loss
        flat_losses = torch.nn.functional.cross_entropy(target_logits, target_ids, reduction="none")

        if not sample_mean:
            return flat_losses.mean()

        # scatter losses back to the original shape and compute sample-mean
        loss_matrix = torch.zeros_like(target_mask, dtype=flat_losses.dtype)
        loss_matrix[target_mask] = flat_losses
        loss = torch.mean(loss_matrix.sum(dim=-1) / target_mask.sum(dim=-1))
        return loss

    def optim_step(self, data: dict[str, list[Any]], epoch_num: int, batch_num: int, step_num: int) -> float | None:
        self.optimizer.zero_grad()

        # construct input conversations
        input_text, target_text = data["prompt"], data["target"]
        conversations = [[{"role": "user", "content": prm}] for prm in input_text]
        conversations = self.adv_model.inject_tokens(conversations)
        encodings = self.adv_model.tokenize(conversations, target_text)

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            result = self.adv_model.forward(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
                adv_mask=encodings.adv_mask,
                adv_embeds=self.univ_embeds,
            )

            loss = self.criterion(
                logits=result.logits,
                input_ids=encodings.input_ids,
                target_mask=encodings.target_mask,
                sample_mean=False,  # NOTE: in the official impl. the average is over all tokens, not over samples
            )

        # grad step
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return loss.item()
