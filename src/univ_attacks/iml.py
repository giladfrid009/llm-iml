from src.sample_attack import SampleAttack
from src.adv_model import AdvModel
from src.activ_extractor import ActivationExtractor, ActivationLoss
from src.eval.evaluator import Evaluator
from src.config import GenConfig
from src.univ_attack import UnivAttack

from typing import Any
import torch
from typing import Iterable, Callable


# NOTE: currently loss is over all target tokens and not a single token per sample
def cosine_similarity_loss(
    univ_activ: torch.Tensor,
    sample_activ: torch.Tensor,
    univ_mask: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        univ_activ (torch.Tensor): Universal activations of shape (batch_size, seq1, hidden_dim).
        sample_activ (torch.Tensor): Sample activations of shape (batch_size, seq2, hidden_dim).
        univ_mask (torch.Tensor): Mask indicating univ tokens are targets, of shape (batch_size, seq1).
        sample_mask (torch.Tensor): Mask indicating sample tokens are targets, of shape (batch_size, seq2).
    """
    univ_mask = univ_mask.bool()
    sample_mask = sample_mask.bool()

    # extract only targets
    univ_targets = univ_activ[univ_mask].reshape(-1, univ_activ.size(-1))
    sample_targets = sample_activ[sample_mask].reshape(-1, sample_activ.size(-1))

    # compute token-wise loss
    flat_losses = 1 - torch.cosine_similarity(univ_targets, sample_targets, dim=-1)

    # scatter losses back to the original shape and compute sample-mean
    sample_losses = torch.zeros_like(sample_mask, dtype=flat_losses.dtype)
    sample_losses[sample_mask] = flat_losses
    return sample_losses.sum(dim=-1) / sample_mask.sum(dim=-1)


# TODO: add scheduling to the inner-attack i.e accept a lambda function that takes the current epoch and
# returns an instance of an inner attack.
# TODO: implement discretization
class IML(UnivAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        internal_attack: SampleAttack,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        activ_extractor: ActivationExtractor,
        evaluators: list[Evaluator],
        eval_freq: int | float = 1,
        mixed_precision: bool = True,
        gen_config: GenConfig | None = None,
        skip_already_fooled: bool = False,
        skip_failed_attacks: bool = True,
        dynamic_labels: bool = False,
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

        self.internal_attack = internal_attack
        self.activ_extractor = activ_extractor
        self.optimizer = optim_factory([self.univ_embeds])
        self.skip_already_fooled = skip_already_fooled
        self.skip_failed_attacks = skip_failed_attacks
        self.dynamic_labels = dynamic_labels  # TODO: implement dynamic labels

        # TODO: (low priority) think of a better, less messy way to register hparams
        self.logger.register_hparams({"iml/internal_attack": self.internal_attack.__class__.__name__})
        self.logger.register_hparams({"iml/optimizer": self.optimizer.__class__.__name__})
        self.logger.register_hparams({"iml/skip_already_fooled": self.skip_already_fooled})
        self.logger.register_hparams({"iml/skip_failed_attacks": self.skip_failed_attacks})
        self.logger.register_hparams({"iml/dynamic_labels": self.dynamic_labels})

        self.logger.register_hparams(activ_extractor.get_hparams())
        self.logger.register_hparams({f"internal_attack/{k}": v for k, v in internal_attack.__dict__.items()})
        self.logger.register_hparams({f"optim/name": self.optimizer.__class__.__name__})
        self.logger.register_hparams({f"optim/{k}": v for k, v in self.optimizer.param_groups[0].items()})

    def optim_step(self, data: dict[str, list[Any]], epoch_num: int, batch_num: int) -> float | None:
        self.optimizer.zero_grad()

        # construct input conversations
        input_text, target_text = data["prompt"], data["target"]
        input_convs = [[{"role": "user", "content": prm}] for prm in input_text]

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            # skip already successfully fooled samples
            if self.skip_already_fooled:
                with torch.inference_mode():
                    responses = self.adv_model.chat(input_convs, self.gen_config)
                    eval_result = self.judge.eval_batch(input_text, responses)
                    mask_fooled = eval_result >= 1.0

                    if mask_fooled.all():
                        return None

                    input_text = [txt for txt, m in zip(input_text, mask_fooled) if not m]
                    input_convs = [conv for conv, m in zip(input_convs, mask_fooled) if not m]
                    target_text = [tgt for tgt, m in zip(target_text, mask_fooled) if not m]

            # run per-sample attack
            with torch.autocast(device_type=self.device.type, enabled=False):
                init_embeds = self.univ_embeds.expand(len(input_convs), -1, -1)
                attack_result = self.internal_attack.fit(input_convs, target_text, init_embeds=init_embeds)
                sample_convs = attack_result.conversations
                sample_embeds = attack_result.adv_embeds

            # skip failed per-sample attacks
            if self.skip_failed_attacks:
                with torch.inference_mode():
                    responses = self.adv_model.chat(sample_convs, self.gen_config, adv_embeds=sample_embeds)
                    eval_result = self.judge.eval_batch(input_text, responses)
                    mask_succ = eval_result >= 1.0

                    if not mask_succ.any():
                        return None

                    sample_embeds = sample_embeds[mask_succ] if sample_embeds is not None else None
                    input_convs = [conv for conv, m in zip(input_convs, mask_succ) if m]
                    sample_convs = [conv for conv, m in zip(sample_convs, mask_succ) if m]
                    target_text = [tgt for tgt, m in zip(target_text, mask_succ) if m]

            with self.activ_extractor.capture():
                # compute per-sample activations
                sample_tokens = self.adv_model.tokenize(sample_convs, target_text)
                with torch.inference_mode():
                    self.adv_model.forward(
                        input_ids=sample_tokens["input_ids"],
                        attention_mask=sample_tokens["attention_mask"],
                        adv_mask=sample_tokens["adv_mask"],
                        adv_embeds=sample_embeds,
                    )
                    sample_activs = self.activ_extractor.get_activations()

                # compute universal activations
                univ_tokens = self.adv_model.tokenize(input_convs, target_text)
                self.adv_model.forward(
                    input_ids=univ_tokens["input_ids"],
                    attention_mask=univ_tokens["attention_mask"],
                    adv_mask=univ_tokens["adv_mask"],
                )
                univ_activs = self.activ_extractor.get_activations()

            # align target mask and activations
            univ_mask = univ_tokens["target_mask"][:, 1:]
            sample_mask = sample_tokens["target_mask"][:, 1:]
            sample_activs = {k: v[:, :-1] for k, v in sample_activs.items()}
            univ_activs = {k: v[:, :-1] for k, v in univ_activs.items()}

            # compute loss
            criterion = ActivationLoss(loss_fn=cosine_similarity_loss)
            loss = criterion.forward(univ_activs, sample_activs, univ_mask=univ_mask, sample_mask=sample_mask)

        # grad step
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return loss.item()
