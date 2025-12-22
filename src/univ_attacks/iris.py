from __future__ import annotations

from src.univ_attacks.univ_attack import TrainPosition
from src.activ_extractor import ActivationLoss
from src.adv_model import AdvModel
from src.eval.evaluator import Evaluator
from src.config import GenConfig
from src.univ_attacks.soft_prompt import SoftPrompt, ce_criterion
from src.fgsm_optim import FGSM
from src.metric_logger import MetricLogger

from typing import Any
import torch
import json
import os


class RefusalConfig:
    def __init__(
        self,
        layer_index: int,
        token_index: int,
        direction_path: str | None = None,
        direction: torch.Tensor | None = None,
    ):
        if direction is None and direction_path is not None:
            direction = torch.load(direction_path, weights_only=True)

        if direction is None:
            raise ValueError("Either direction_path or direction must be provided.")

        assert token_index < 0, "Token index must be negative."
        assert direction.ndim == 1, "Direction must be a 1D tensor."

        self.layer_index = layer_index
        self.token_index = token_index
        self.direction_path = direction_path
        self.direction = direction

    def get_hparams(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "token_index": self.token_index,
            "direction_shape": self.direction.shape,
            "direction_path": self.direction_path,
        }

    def save(self, path: str, include_direction: bool | None = None) -> None:
        if include_direction is None:
            include_direction = path.endswith(".pt") or path.endswith(".pth")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if include_direction:
            if self.direction is None:
                raise ValueError("direction is not available to save.")

            data = {
                "layer_index": self.layer_index,
                "token_index": self.token_index,
                "direction": self.direction.detach().cpu(),
                "direction_path": self.direction_path,
            }

            torch.save(data, f=path)

        else:
            if self.direction_path is None:
                raise ValueError("direction_path is not available to save.")

            data = {
                "layer_index": self.layer_index,
                "token_index": self.token_index,
                "direction_path": self.direction_path,
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=4)

    @classmethod
    def load(cls, path: str, include_direction: bool | None = None) -> RefusalConfig:
        if include_direction is None:
            include_direction = path.endswith(".pt") or path.endswith(".pth")

        if include_direction:
            data = torch.load(path)
            if not isinstance(data, dict):
                raise ValueError("Invalid data format in the saved file.")

            return cls(
                layer_index=data["layer_index"],
                token_index=data["token_index"],
                direction=data["direction"],
                direction_path=data.get("direction_path", None),
            )

        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            layer_index=data["layer_index"],
            token_index=data["token_index"],
            direction_path=data["direction_path"],
        )


def iris_criterion(
    activs: torch.Tensor,
    token_index: int,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        activs (torch.Tensor): Activations of shape (batch_size, seq_len, hidden_size).
        token_index (int): Token index at which the loss is computed.
        direction (torch.Tensor): Refusal direction tensor, of shape (hidden_size)

    Returns:
        torch.Tensor: Computed loss for each sample, of shape (batch_size,).
    """
    assert direction.ndim == 1, "Direction must be a 1D tensor."
    assert direction.size(0) == activs.size(2), "Direction size must match hidden size."
    direction = direction.to(activs.device, dtype=activs.dtype)

    token_activs = activs[:, token_index]  # (B, H)
    loss = torch.square(token_activs @ direction)
    return loss


class IRIS(SoftPrompt):
    """
    IRIS attack implemented on top of SoftPrompt attack. The proposed IRIS loss is augmenting the standard cross-entropy loss
    with an activation-based loss that encourages the model's activations to deviate from refusal directions.

    - Stronger Universal and Transferable Attacks by Suppressing Refusals: [https://aclanthology.org/2025.naacl-long.302.pdf]
    """

    def __init__(
        self,
        adv_model: AdvModel,
        optimizer: FGSM,
        refusal_config: RefusalConfig,
        evaluators: list[Evaluator],
        beta: float = 0.5,
        eval_metric: str | None = None,
        eval_freq: int | float = 1,
        mixed_precision: bool = False,
        gen_config: GenConfig | None = None,
        metric_logger: MetricLogger | None = None,
    ):
        super().__init__(
            adv_model=adv_model,
            optimizer=optimizer,
            evaluators=evaluators,
            eval_metric=eval_metric,
            eval_freq=eval_freq,
            mixed_precision=mixed_precision,
            gen_config=gen_config,
            metric_logger=metric_logger,
        )

        self.refusal_config = refusal_config
        self.beta = beta

        self.metric_logger.report_hparams("attack", beta=self.beta, refusal_layer=refusal_config.layer_index)
        self.metric_logger.report_hparams("refusal_config", self.refusal_config.get_hparams())

        # move direction to device
        self.refusal_config.direction = self.refusal_config.direction.to(self.device)

    def optim_step(self, data: dict[str, list[Any]], position: TrainPosition) -> dict[str, float | None]:
        self.optimizer.zero_grad()

        # construct input conversations
        input_text, target_text = data["prompt"], data["target"]
        conversations = [[{"role": "user", "content": prm}] for prm in input_text]
        conversations = self.adv_model.inject_tokens(conversations)

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            full_encodings = self.adv_model.tokenize(conversations, target_text)
            input_encodings = self.adv_model.tokenize(conversations)

            # use full encodings to compute soft-prompt loss over target tokens
            full_result = self.adv_model.forward(
                input_ids=full_encodings.input_ids,
                attention_mask=full_encodings.attention_mask,
                adv_mask=full_encodings.adv_mask,
                adv_embeds=self.univ_embeds,
            )

            ce_loss = ce_criterion(
                logits=full_result.logits,
                input_ids=full_encodings.input_ids,
                target_mask=full_encodings.target_mask,
                sample_mean=False,
            )

            hidden_states: tuple[torch.Tensor] = self.adv_model.forward(
                input_ids=input_encodings.input_ids,
                attention_mask=input_encodings.attention_mask,
                adv_mask=input_encodings.adv_mask,
                adv_embeds=self.univ_embeds,
                output_hidden_states=True,
            ).hidden_states

            # each hidden state of shape (batch_size, seq_len, hidden_size)
            activs = {f"state-{i}": hidden_states[i] for i in range(1, len(hidden_states))}

            # iris loss is computed w.r.t the last tokens of the input prompt
            iris_loss = ActivationLoss(loss_fn=iris_criterion, reduction="sum-mean").forward(
                activs,
                token_index=self.refusal_config.token_index,  # NOTE: in the paper they use "last input index"
                direction=self.refusal_config.direction,
            )

            loss = (1 - self.beta) * ce_loss + self.beta * iris_loss

        # grad step
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {
            "loss": loss.item(),
            "loss/iris_loss": iris_loss.item() * self.beta,
            "loss/ce_loss": ce_loss.item() * (1 - self.beta),
        }
