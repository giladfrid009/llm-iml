from src.adv_model import AdvModel
from src.eval.evaluator import Evaluator
from src.config import GenConfig
from src.univ_attacks.univ_attack import UnivAttack, TrainPosition
from src.fgsm_optim import FGSM
from src.utils.trackers import MetricTracker

from typing import Any
import torch


def ce_criterion(
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
    loss_grid = torch.zeros_like(target_mask, dtype=flat_losses.dtype)
    loss_grid[target_mask] = flat_losses
    counts = target_mask.sum(dim=-1).float().clamp_min(1.0)
    loss = torch.mean(loss_grid.sum(dim=-1) / counts)
    return loss


class SoftPrompt(UnivAttack):
    """
    Universal variant of the Soft Prompt attack.
    - Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space [https://arxiv.org/abs/2402.09063]
    """
    def __init__(
        self,
        adv_model: AdvModel,
        optimizer: FGSM,
        evaluators: list[Evaluator],
        eval_metric: str | None = None,
        eval_freq: int | float = 1,
        mixed_precision: bool = False,
        gen_config: GenConfig | None = None,
        metric_tracker: MetricTracker | None = None,
    ):
        super().__init__(
            adv_model=adv_model,
            evaluators=evaluators,
            eval_metric=eval_metric,
            eval_freq=eval_freq,
            mixed_precision=mixed_precision,
            gen_config=gen_config,
            metric_tracker=metric_tracker,
        )

        self.optimizer = optimizer

        self.metric_tracker.report_hparams("attack", optimizer=type(self.optimizer).__name__)
        self.metric_tracker.report_hparams("optim", optimizer.state_dict()["param_groups"][0], name=type(self.optimizer).__name__)

    def optim_step(self, data: dict[str, list[Any]], position: TrainPosition) -> dict[str, float | None]:
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

            loss = ce_criterion(
                logits=result.logits,
                input_ids=encodings.input_ids,
                target_mask=encodings.target_mask,
                sample_mean=False,  # NOTE: in the official impl. the average is over all tokens, not over samples
            )

        # grad step
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {"loss": loss.item()}
