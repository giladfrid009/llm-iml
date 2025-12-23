from src.sample_attacks import SampleAttack
from src.adv_model import AdvModel
from src.eval.evaluator import Evaluator
from src.config import GenConfig
from src.univ_attacks.univ_attack import UnivAttack, TrainPosition
from src.utils.trackers import MetricTracker

import inspect
from typing import Any, Callable
import torch


class UAP(UnivAttack):
    """
    A batched variant of the classic approach for constructing UAPs, based on per-sample attacks.
    *NOTE:* To perform original UAP algorithm, the batch size should be set to 1.

    - Universal adversarial perturbations: [https://arxiv.org/abs/1610.08401]
    - Universal Adversarial Perturbations for Speech Recognition Systems: [https://arxiv.org/abs/1905.03828]
    - Universal Adversarial Audio Perturbations [https://arxiv.org/abs/1908.03173]
    """

    def __init__(
        self,
        adv_model: AdvModel,
        inner_attack: SampleAttack | Callable[[AdvModel, int], SampleAttack],
        evaluators: list[Evaluator],
        eval_metric: str | None = None,
        eval_freq: int | float = 1,
        mixed_precision: bool = False,
        gen_config: GenConfig | None = None,
        skip_already_fooled: bool = True,
        skip_failed_attacks: bool = True,
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

        if callable(inner_attack):
            self.attack_builder_func = inner_attack
            self.inner_attack = self.attack_builder_func(adv_model, 0)
        else:
            self.attack_builder_func = None
            self.inner_attack = inner_attack

        self.skip_already_fooled = skip_already_fooled
        self.skip_failed_attacks = skip_failed_attacks

        self.metric_tracker.report_hparams(
            "attack",
            inner_attack=type(self.inner_attack).__name__,
            attack_builder=inspect.getsource(self.attack_builder_func) if self.attack_builder_func else None,
            skip_already_fooled=self.skip_already_fooled,
            skip_failed_attacks=self.skip_failed_attacks,
        )

        self.metric_tracker.report_hparams("inner_attack", self.inner_attack.get_hparams())

    @property
    def judge_evaluator(self) -> Evaluator:
        """
        Returns the evaluator used for judging the success of the attack.
        """
        for ev in self.evaluators:
            if self.eval_metric in ev.metric_names:
                return ev

        raise ValueError(
            f"Judge metric {self.eval_metric} not found in any evaluator. Available metrics: {[ev.metric_names for ev in self.evaluators]}"
        )

    def make_attack(self, epoch_num: int) -> SampleAttack:
        if self.attack_builder_func is None:
            return self.inner_attack
        return self.attack_builder_func(self.adv_model, epoch_num)

    def optim_step(self, data: dict[str, list[Any]], position: TrainPosition) -> dict[str, float | None]:
        # create new instance of inner attack for each epoch
        if position.epoch > 0 and position.batch == 0:
            self.inner_attack = self.make_attack(position.epoch)

        METRICS = {}

        # construct input conversations
        input_texts, target_texts = data["prompt"], data["target"]
        input_convs = [[{"role": "user", "content": prm}] for prm in input_texts]
        input_convs = self.adv_model.inject_tokens(input_convs)

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            # skip already successfully fooled samples
            if self.skip_already_fooled:
                with torch.inference_mode():
                    init_responses = self.adv_model.chat(
                        conversations=input_convs,
                        adv_embeds=self.univ_embeds,
                        config=self.gen_config,
                    )

                    eval_result = self.judge_evaluator.eval_batch(input_texts, init_responses)
                    eval_metric = torch.tensor(eval_result[self.eval_metric], device=self.device)
                    fooled_mask = eval_metric >= 1.0

                    not_fooled_ratio = 1 - fooled_mask.float().mean().item()
                    METRICS["UAP/not_fooled_ratio"] = not_fooled_ratio

                    if fooled_mask.all():  # all samples already fooled
                        METRICS["UAP/effective_batch_ratio"] = 0.0
                        METRICS["loss"] = None
                        return METRICS

                    input_texts = [txt for txt, m in zip(input_texts, fooled_mask) if not m]
                    input_convs = [conv for conv, m in zip(input_convs, fooled_mask) if not m]
                    target_texts = [tgt for tgt, m in zip(target_texts, fooled_mask) if not m]

            # run per-sample attack
            with torch.autocast(device_type=self.device.type, enabled=False):
                init_embeds = self.univ_embeds.repeat(len(input_convs), 1, 1)
                clean_convs = [[{"role": "user", "content": prm}] for prm in input_texts]
                sample_result = self.inner_attack.fit(clean_convs, target_texts, init_embeds=init_embeds)

                if sample_losses := sample_result.logs.get("loss"):
                    METRICS["UAP/initial_sample_attack_loss"] = sample_losses[0]

            # skip failed per-sample attacks
            if self.skip_failed_attacks:
                with torch.inference_mode():
                    sample_responses = self.adv_model.chat(
                        conversations=sample_result.conversations,
                        adv_embeds=sample_result.adv_embeds,
                        config=self.gen_config,
                    )

                    eval_result = self.judge_evaluator.eval_batch(input_texts, sample_responses)
                    eval_metric = torch.tensor(eval_result[self.eval_metric], device=self.device)
                    success_mask = eval_metric >= 1.0

                    sample_asr = success_mask.float().mean().item()
                    METRICS["UAP/sample_attack_success_ratio"] = sample_asr

                    if not success_mask.any():  # all attacks failed
                        METRICS["UAP/effective_batch_ratio"] = 0.0
                        METRICS["loss"] = None
                        return METRICS

                    input_convs = [conv for conv, m in zip(input_convs, success_mask) if m]
                    target_texts = [tgt for tgt, m in zip(target_texts, success_mask) if m]
                    sample_result = sample_result.masked_select(success_mask)

            batch_ratio = len(input_convs) / len(data["prompt"])
            METRICS["UAP/effective_batch_ratio"] = batch_ratio

        if sample_result.adv_embeds is None:
            raise ValueError("Inner attack did not return adversarial embeddings.")

        # update universal perturbation
        delta = torch.sum(sample_result.adv_embeds - init_embeds, dim=0, keepdim=True)
        self.univ_embeds.add_(delta)

        METRICS["loss"] = delta.abs().mean().item()
        return METRICS
