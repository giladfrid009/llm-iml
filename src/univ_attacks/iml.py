from src.sample_attacks.sample_attack import SampleAttack
from src.adv_model import AdvModel
from src.activ_extractor import ActivationExtractor, ActivationLoss
from src.eval.evaluator import Evaluator
from src.config import GenConfig
from src.univ_attacks.univ_attack import UnivAttack
from src.metric_logger import MetricLogger

import inspect
from typing import Any, Callable
import torch


# NOTE: currently loss is over all target tokens and not a single token per sample
def cosine_similarity_loss(
    univ_activ: torch.Tensor,
    sample_activ: torch.Tensor,
    univ_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    sample_mean: bool = True,
) -> torch.Tensor:
    """
    Args:
        univ_activ (torch.Tensor): Universal activations of shape (batch_size, seq1, hidden_dim).
        sample_activ (torch.Tensor): Sample activations of shape (batch_size, seq2, hidden_dim).
        univ_mask (torch.Tensor): Mask indicating univ tokens are targets, of shape (batch_size, seq1).
        sample_mask (torch.Tensor): Mask indicating sample tokens are targets, of shape (batch_size, seq2).
        sample_mean (bool): Averaging method of the loss
            - If True, first average over all target tokens for each sample, then average over samples.
            - If False, overall loss is average over all target tokens across all samples.
    """
    univ_mask = univ_mask.bool()
    sample_mask = sample_mask.bool()

    # align masks and activations
    univ_mask = univ_mask[:, 1:]  # remove BOS token
    sample_mask = sample_mask[:, 1:]  # remove BOS token
    univ_activ = univ_activ[:, :-1]  # remove new token
    sample_activ = sample_activ[:, :-1]  # remove new token

    # extract only targets
    univ_targets = univ_activ[univ_mask].reshape(-1, univ_activ.size(-1))
    sample_targets = sample_activ[sample_mask].reshape(-1, sample_activ.size(-1))

    # compute token-wise loss
    flat_losses = 1 - torch.cosine_similarity(univ_targets, sample_targets, dim=-1)

    if not sample_mean:
        return flat_losses.mean()

    # scatter losses back to the original shape and compute sample-mean
    sample_losses = torch.zeros_like(sample_mask, dtype=flat_losses.dtype)
    sample_losses[sample_mask] = flat_losses
    return sample_losses.sum(dim=-1) / sample_mask.sum(dim=-1)


class IML(UnivAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        inner_attack: SampleAttack | Callable[[AdvModel, int], SampleAttack],
        optimizer: torch.optim.Optimizer,
        activ_extractor: ActivationExtractor,
        evaluators: list[Evaluator],
        judge_metric: str | None = None,
        eval_freq: int | float = 1,
        mixed_precision: bool = True,
        gen_config: GenConfig | None = None,
        skip_already_fooled: bool = False,
        skip_failed_attacks: bool = True,
        dynamic_labels: int = -1,
        metric_logger: MetricLogger | None = None,
    ):
        super().__init__(
            adv_model=adv_model,
            evaluators=evaluators,
            judge_metric=judge_metric,
            eval_freq=eval_freq,
            mixed_precision=mixed_precision,
            gen_config=gen_config,
            metric_logger=metric_logger,
        )

        if callable(inner_attack):
            self.attack_builder_func = inner_attack
            self.inner_attack = self.attack_builder_func(adv_model, 0)
        else:
            self.attack_builder_func = None
            self.inner_attack = inner_attack

        self.activ_extractor = activ_extractor
        self.optimizer = optimizer
        self.skip_already_fooled = skip_already_fooled
        self.skip_failed_attacks = skip_failed_attacks
        self.dynamic_labels = dynamic_labels

        self.metric_logger.log_hparams(
            "iml",
            inner_attack=self.inner_attack.__class__.__name__,
            attack_builder=inspect.getsource(self.attack_builder_func) if self.attack_builder_func else None,
            optimizer=self.optimizer.__class__.__name__,
            skip_already_fooled=self.skip_already_fooled,
            skip_failed_attacks=self.skip_failed_attacks,
            dynamic_labels=self.dynamic_labels,
        )

        if self.attack_builder_func is not None:
            self.metric_logger.log_hparams("iml", attack_builder=inspect.getsource(self.attack_builder_func))

        self.metric_logger.log_hparams("activ_extractor", activ_extractor.get_hparams())
        self.metric_logger.log_hparams("inner_attack", self.inner_attack.get_hparams())
        self.metric_logger.log_hparams(
            "optim", optimizer.state_dict()["param_groups"][0], name=self.optimizer.__class__.__name__
        )

    @property
    def judge_evaluator(self) -> Evaluator:
        """
        Returns the evaluator used for judging the success of the attack.
        """
        for ev in self.evaluators:
            if self.judge_metric in ev.metric_names:
                return ev
        raise ValueError(f"Judge metric {self.judge_metric} not found in any evaluator.")

    def make_attack(self, epoch_num: int) -> SampleAttack:
        if self.attack_builder_func is None:
            return self.inner_attack
        return self.attack_builder_func(self.adv_model, epoch_num)

    def truncate_tokens(self, sequences: list[str], n: int) -> list[str]:
        encoding = self.adv_model.tokenizer(
            sequences,
            add_special_tokens=False,
            truncation=True,
            max_length=n,
        )
        return self.adv_model.tokenizer.batch_decode(
            encoding.input_ids,
            skip_special_tokens=True,
        )

    def optim_step(self, data: dict[str, list[Any]], epoch_num: int, batch_num: int) -> float | None:
        # create new instance of inner attack for each epoch
        if epoch_num > 0 and batch_num == 0:
            self.inner_attack = self.make_attack(epoch_num)

        self.optimizer.zero_grad()

        # construct input conversations
        input_texts, target_texts = data["prompt"], data["target"]
        input_convs = [[{"role": "user", "content": prm}] for prm in input_texts]

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            # skip already successfully fooled samples
            if self.skip_already_fooled:
                with torch.inference_mode():
                    responses = self.adv_model.chat(input_convs, self.gen_config)
                    eval_result = self.judge_evaluator.eval_batch(input_texts, responses)
                    eval_metric = torch.tensor(eval_result[self.judge_metric], device=self.device)
                    fooled_mask = eval_metric >= 1.0

                    if fooled_mask.all():
                        return None

                    input_texts = [txt for txt, m in zip(input_texts, fooled_mask) if not m]
                    input_convs = [conv for conv, m in zip(input_convs, fooled_mask) if not m]
                    target_texts = [tgt for tgt, m in zip(target_texts, fooled_mask) if not m]

            # run per-sample attack
            with torch.autocast(device_type=self.device.type, enabled=False):
                init_embeds = self.univ_embeds.expand(len(input_convs), -1, -1)
                attack_result = self.inner_attack.fit(input_convs, target_texts, init_embeds=init_embeds)
                sample_convs = attack_result.conversations
                sample_embeds = attack_result.adv_embeds

            # skip failed per-sample attacks
            if self.skip_failed_attacks:
                with torch.inference_mode():
                    responses = self.adv_model.chat(sample_convs, self.gen_config, adv_embeds=sample_embeds)
                    eval_result = self.judge_evaluator.eval_batch(input_texts, responses)
                    eval_metric = torch.tensor(eval_result[self.judge_metric], device=self.device)
                    success_mask = eval_metric >= 1.0

                    if not success_mask.any():
                        return None

                    # set target texts to generated responses
                    if self.dynamic_labels > 0:
                        target_texts = self.truncate_tokens(responses, self.dynamic_labels)

                    input_convs = [conv for conv, m in zip(input_convs, success_mask) if m]
                    sample_convs = [conv for conv, m in zip(sample_convs, success_mask) if m]
                    target_texts = [tgt for tgt, m in zip(target_texts, success_mask) if m]
                    sample_embeds = sample_embeds[success_mask] if sample_embeds is not None else None

            elif self.dynamic_labels > 0 and (not self.skip_failed_attacks):
                # we first need to generate responses
                responses = self.adv_model.chat(
                    sample_convs,
                    self.gen_config,
                    adv_embeds=sample_embeds,
                    max_new_tokens=self.dynamic_labels,
                )

                # set target texts to generated responses
                target_texts = self.truncate_tokens(responses, self.dynamic_labels)

            with self.activ_extractor.capture():
                # compute per-sample activations
                sample_encodings = self.adv_model.tokenize(sample_convs, target_texts)
                with torch.inference_mode():
                    self.adv_model.forward(
                        input_ids=sample_encodings.input_ids,
                        attention_mask=sample_encodings.attention_mask,
                        adv_mask=sample_encodings.adv_mask,
                        adv_embeds=sample_embeds,
                    )
                    sample_activs = self.activ_extractor.get_activations()

                # compute universal activations
                univ_encodings = self.adv_model.tokenize(input_convs, target_texts)
                self.adv_model.forward(
                    input_ids=univ_encodings.input_ids,
                    attention_mask=univ_encodings.attention_mask,
                    adv_mask=univ_encodings.adv_mask,
                )
                univ_activs = self.activ_extractor.get_activations()

            # compute loss
            criterion = ActivationLoss(loss_fn=cosine_similarity_loss)
            loss = criterion.forward(
                univ_activs,
                sample_activs,
                univ_mask=univ_encodings.target_mask,
                sample_mask=sample_encodings.target_mask,
                sample_mean=False,
            )

        # grad step
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return loss.item()
