from src.sample_attacks import SampleAttack
from src.adv_model import AdvModel
from src.activ_extractor import ActivationExtractor, ActivationLoss
from src.eval.evaluator import Evaluator
from src.config import GenConfig
from src.univ_attacks.univ_attack import UnivAttack, TrainPosition
from src.utils.trackers import MetricTracker

import inspect
from typing import Any, Callable
import torch
import torch.nn.functional as F


def compute_iap_uap_sim(
    univ_activs: dict[str, torch.Tensor],
    sample_activs: dict[str, torch.Tensor],
    univ_mask: torch.Tensor,
    sample_mask: torch.Tensor,
) -> float:
    # Slicing to remove BOS
    u_mask = univ_mask[:, 1:].bool()
    s_mask = sample_mask[:, 1:].bool()

    iap_uap_sims = []

    for layer_name in univ_activs:
        u_act = univ_activs[layer_name][:, :-1]
        s_act = sample_activs[layer_name][:, :-1]

        u_flat = u_act[u_mask].reshape(-1, u_act.size(-1))
        s_flat = s_act[s_mask].reshape(-1, s_act.size(-1))

        tok_sims = torch.cosine_similarity(u_flat, s_flat, dim=-1)

        grid = torch.zeros(s_mask.shape, device=s_mask.device, dtype=tok_sims.dtype)
        grid[s_mask] = tok_sims

        counts = s_mask.sum(dim=-1).float().clamp(min=1.0)
        sample_avg_sim = grid.sum(dim=-1) / counts
        iap_uap_sims.append(sample_avg_sim)

    if not iap_uap_sims:
        return 0.0

    # Avg over layers for each sample, then avg over samples
    return torch.stack(iap_uap_sims).mean(dim=0).mean().item()


def compute_iap_iap_sim(
    sample_activs: dict[str, torch.Tensor],
    sample_mask: torch.Tensor,
) -> float:
    # Slicing to remove BOS
    s_mask = sample_mask[:, 1:].bool()

    iap_iap_sims = []

    for layer_name in sample_activs:
        s_act = sample_activs[layer_name][:, :-1]

        # Mean pool per sample
        counts = s_mask.sum(dim=-1).float().clamp(min=1.0)
        s_grid = torch.zeros(s_mask.shape + (s_act.size(-1),), device=s_act.device, dtype=s_act.dtype)
        s_grid[s_mask] = s_act[s_mask]  # flatten assignment
        s_means = s_grid.sum(dim=1) / counts.unsqueeze(-1)

        s_norm = F.normalize(s_means, p=2, dim=1)
        sim_mat = torch.mm(s_norm, s_norm.t())

        B = sim_mat.size(0)
        if B > 1:
            off_diag = ~torch.eye(B, device=sim_mat.device, dtype=torch.bool)
            iap_iap_sims.append(sim_mat[off_diag].mean().item())
        else:
            iap_iap_sims.append(0.0)

    # Aggregate (Avg over layers)
    if not iap_iap_sims:
        return 0.0

    return sum(iap_iap_sims) / len(iap_iap_sims)


def compute_similarity_metrics(
    univ_activs: dict[str, torch.Tensor],
    sample_activs: dict[str, torch.Tensor],
    univ_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    sample_success_mask: torch.Tensor,
    prefix: str = "IML",
) -> dict[str, float]:
    metrics = {}

    # --- 1. IAP vs UAP ---
    # A. All samples
    metrics[f"{prefix}/iap_uap_sim_all"] = compute_iap_uap_sim(univ_activs, sample_activs, univ_mask, sample_mask)

    # B. Successful samples
    if sample_success_mask.any():
        # filter inputs
        u_activs_succ = {k: v[sample_success_mask] for k, v in univ_activs.items()}
        s_activs_succ = {k: v[sample_success_mask] for k, v in sample_activs.items()}
        u_mask_succ = univ_mask[sample_success_mask]
        s_mask_succ = sample_mask[sample_success_mask]

        metrics[f"{prefix}/iap_uap_sim_success"] = compute_iap_uap_sim(u_activs_succ, s_activs_succ, u_mask_succ, s_mask_succ)

    # --- 2. IAP vs IAP ---
    # A. All samples
    metrics[f"{prefix}/iap_iap_sim_all"] = compute_iap_iap_sim(sample_activs, sample_mask)

    # B. Successful samples
    if sample_success_mask.sum() > 1:
        s_activs_succ = {k: v[sample_success_mask] for k, v in sample_activs.items()}
        s_mask_succ = sample_mask[sample_success_mask]

        metrics[f"{prefix}/iap_iap_sim_success"] = compute_iap_iap_sim(s_activs_succ, s_mask_succ)
    elif sample_success_mask.any():
        metrics[f"{prefix}/iap_iap_sim_success"] = 0.0

    return metrics


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

    Returns:
        torch.Tensor: Computed loss for each sample, of shape (batch_size,).
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
        scalar_loss = flat_losses.mean()
        return scalar_loss.expand(sample_mask.size(0))  # expand to batch size

    # scatter losses back to the original shape and compute sample-mean
    sample_losses = torch.zeros_like(sample_mask, dtype=flat_losses.dtype)
    sample_losses[sample_mask] = flat_losses
    return sample_losses.sum(dim=-1) / sample_mask.sum(dim=-1)


class IML_Extra(UnivAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        inner_attack: SampleAttack | Callable[[AdvModel, int], SampleAttack],
        optimizer: torch.optim.Optimizer,
        activ_extractor: ActivationExtractor,
        evaluators: list[Evaluator],
        eval_metric: str | None = None,
        eval_freq: int | float = 1,
        mixed_precision: bool = False,
        gen_config: GenConfig | None = None,
        skip_already_fooled: bool = True,
        skip_failed_attacks: bool = True,
        warmup_epochs: int = 0,
        dynamic_labels: int = -1,
        target_controls: bool = False,
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

        self.activ_extractor = activ_extractor
        self.optimizer = optimizer
        self.skip_already_fooled = skip_already_fooled
        self.skip_failed_attacks = skip_failed_attacks
        self.warmup_epochs = warmup_epochs
        self.dynamic_labels = dynamic_labels
        self.target_controls = target_controls

        self.metric_tracker.report_hparams(
            "attack",
            inner_attack=type(self.inner_attack).__name__,
            attack_builder=inspect.getsource(self.attack_builder_func) if self.attack_builder_func else None,
            optimizer=type(self.optimizer).__name__,
            skip_already_fooled=self.skip_already_fooled,
            skip_failed_attacks=self.skip_failed_attacks,
            warmup_epochs=self.warmup_epochs,
            dynamic_labels=self.dynamic_labels,
            target_controls=self.target_controls,
        )

        self.metric_tracker.report_hparams("activ_extractor", activ_extractor.get_hparams())
        self.metric_tracker.report_hparams("inner_attack", self.inner_attack.get_hparams())
        self.metric_tracker.report_hparams("optim", optimizer.state_dict()["param_groups"][0], name=type(self.optimizer).__name__)

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

    def optim_step(self, data: dict[str, list[Any]], position: TrainPosition) -> dict[str, float | None]:
        # create new instance of inner attack for each epoch
        if position.epoch > 0 and position.batch == 0:
            self.inner_attack = self.make_attack(position.epoch)

        METRICS = {}
        self.optimizer.zero_grad()

        # construct input conversations
        input_texts, target_texts = data["prompt"], data["target"]
        input_convs = [[{"role": "user", "content": prm}] for prm in input_texts]
        input_convs = self.adv_model.inject_tokens(input_convs)

        fooled_mask = torch.zeros(len(input_convs), dtype=torch.bool, device=self.device)
        sample_success_mask = torch.ones(len(input_convs), dtype=torch.bool, device=self.device)

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
                    METRICS["IML/not_fooled_ratio"] = not_fooled_ratio
                    METRICS["IML/fooled_ratio"] = 1.0 - not_fooled_ratio

                    # if fooled_mask.all():  # all samples already fooled
                    #     METRICS["IML/effective_batch_ratio"] = 0.0
                    #     METRICS["loss"] = None
                    #     return METRICS

                    # input_texts = [txt for txt, m in zip(input_texts, fooled_mask) if not m]
                    # input_convs = [conv for conv, m in zip(input_convs, fooled_mask) if not m]
                    # target_texts = [tgt for tgt, m in zip(target_texts, fooled_mask) if not m]

            # run per-sample attack
            with torch.autocast(device_type=self.device.type, enabled=False):
                init_embeds = None
                if position.epoch >= self.warmup_epochs:
                    # use universal embeds as init only after warmup
                    init_embeds = self.univ_embeds.repeat(len(input_convs), 1, 1)

                clean_convs = [[{"role": "user", "content": prm}] for prm in input_texts]
                sample_result = self.inner_attack.fit(clean_convs, target_texts, init_embeds=init_embeds)

                if sample_losses := sample_result.logs.get("loss"):
                    METRICS["IML/initial_sample_attack_loss"] = sample_losses[0]
                    METRICS["IML/final_sample_attack_loss"] = sample_losses[-1]

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
                    sample_success_mask = eval_metric >= 1.0

                    sample_asr = sample_success_mask.float().mean().item()
                    METRICS["IML/sample_attack_success_ratio"] = sample_asr

                    # if not sample_success_mask.any():  # all attacks failed
                    #     METRICS["IML/effective_batch_ratio"] = 0.0
                    #     METRICS["loss"] = None
                    #     return METRICS

                    if self.dynamic_labels > 0:
                        # set target texts to generated sample responses
                        target_texts = self.truncate_tokens(sample_responses, self.dynamic_labels)

                    # input_convs = [conv for conv, m in zip(input_convs, sample_success_mask) if m]
                    # target_texts = [tgt for tgt, m in zip(target_texts, sample_success_mask) if m]
                    # sample_result = sample_result.masked_select(sample_success_mask)

            elif self.dynamic_labels > 0 and (not self.skip_failed_attacks):
                # explicitly use all sample responses as new target texts
                sample_responses = self.adv_model.chat(
                    conversations=sample_result.conversations,
                    adv_embeds=sample_result.adv_embeds,
                    config=self.gen_config,
                    max_new_tokens=self.dynamic_labels,
                )

                # set target texts to generated responses
                target_texts = self.truncate_tokens(sample_responses, self.dynamic_labels)

            global_mask = torch.logical_not(fooled_mask) & sample_success_mask

            batch_ratio = global_mask.float().mean().item()
            METRICS["IML/effective_batch_ratio"] = batch_ratio

            with self.activ_extractor.capture():
                # compute per-sample activations
                sample_encodings = self.adv_model.tokenize(
                    sample_result.conversations,
                    target_texts,
                    target_controls=self.target_controls,
                )

                with torch.inference_mode():
                    self.adv_model.forward(
                        input_ids=sample_encodings.input_ids,
                        attention_mask=sample_encodings.attention_mask,
                        adv_mask=sample_encodings.adv_mask,
                        adv_embeds=sample_result.adv_embeds,
                    )
                    sample_activs = self.activ_extractor.get_activations()

                # compute universal activations
                univ_encodings = self.adv_model.tokenize(
                    input_convs,
                    target_texts,
                    target_controls=self.target_controls,
                )

                self.adv_model.forward(
                    input_ids=univ_encodings.input_ids,
                    attention_mask=univ_encodings.attention_mask,
                    adv_mask=univ_encodings.adv_mask,
                    adv_embeds=self.univ_embeds,
                )
                univ_activs = self.activ_extractor.get_activations()

            # COMPUTE VARIOUS METRICS
            with torch.no_grad():
                sim_metrics = compute_similarity_metrics(
                    univ_activs,
                    sample_activs,
                    univ_encodings.target_mask,
                    sample_encodings.target_mask,
                    sample_success_mask,
                )
                METRICS.update(sim_metrics)

            # filter activations to only effective samples
            for key in univ_activs.keys():
                univ_activs[key] = univ_activs[key][global_mask]
                sample_activs[key] = sample_activs[key][global_mask]

            univ_mask = univ_encodings.target_mask[global_mask]
            sample_mask = sample_encodings.target_mask[global_mask]

            # compute loss
            criterion = ActivationLoss(loss_fn=cosine_similarity_loss, reduction="sum-mean")
            loss = criterion.forward(
                univ_activs,
                sample_activs,
                univ_mask=univ_mask,
                sample_mask=sample_mask,
                sample_mean=False,  # NOTE: when sample_mean=True it performs worse
            )

        # grad step
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        METRICS["loss"] = loss.item()
        return METRICS
