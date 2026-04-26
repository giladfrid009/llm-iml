from src.activ_extractor import ActivationLoss
from src.univ_attacks.upd import UPD
from src.univ_attacks.univ_attack import TrainPosition

from typing import Any
import torch


def ila_loss(
    univ_activ: torch.Tensor,
    sample_activ: torch.Tensor,
    clean_activ: torch.Tensor,
    univ_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    clean_mask: torch.Tensor,
    sample_mean: bool = True,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Args:
        univ_activ (torch.Tensor): Universal activations of shape (batch_size, seq1, hidden_dim).
        sample_activ (torch.Tensor): Sample activations of shape (batch_size, seq2, hidden_dim).
        clean_activ (torch.Tensor): Clean activations of shape (batch_size, seq3, hidden_dim).
        univ_mask (torch.Tensor): Mask indicating univ tokens are targets, of shape (batch_size, seq1).
        sample_mask (torch.Tensor): Mask indicating sample tokens are targets, of shape (batch_size, seq2).
        clean_mask (torch.Tensor): Mask indicating clean tokens are targets, of shape (batch_size, seq3).
        sample_mean (bool): Averaging method of the loss
            - If True, first average over all target tokens for each sample, then average over samples.
            - If False, overall loss is average over all target tokens across all samples.

    Returns:
        torch.Tensor: Computed loss for each sample, of shape (batch_size,).
    """
    univ_mask = univ_mask.bool()
    sample_mask = sample_mask.bool()
    clean_mask = clean_mask.bool()

    # align masks and activations
    univ_mask = univ_mask[:, 1:]  # remove BOS token
    sample_mask = sample_mask[:, 1:]  # remove BOS token
    clean_mask = clean_mask[:, 1:]  # remove BOS token
    univ_activ = univ_activ[:, :-1]  # remove new token
    sample_activ = sample_activ[:, :-1]  # remove new token
    clean_activ = clean_activ[:, :-1]  # remove new token

    # extract only targets
    univ_targets = univ_activ[univ_mask].reshape(-1, univ_activ.size(-1))
    sample_targets = sample_activ[sample_mask].reshape(-1, sample_activ.size(-1))
    clean_targets = clean_activ[clean_mask].reshape(-1, clean_activ.size(-1))

    # subtract clean activations from both univ and sample activations to get deltas
    univ_targets = univ_targets - clean_targets
    sample_targets = sample_targets - clean_targets

    # compute token-wise loss
    if normalized:
        flat_losses = 1 - torch.cosine_similarity(univ_targets, sample_targets, dim=-1)
    else:
        flat_losses = -torch.sum(univ_targets * sample_targets, dim=-1)  # negative dot-product

    if not sample_mean:
        scalar_loss = flat_losses.mean()
        return scalar_loss.expand(sample_mask.size(0))  # expand to batch size

    # scatter losses back to the original shape and compute sample-mean
    loss_grid = torch.zeros_like(sample_mask, dtype=flat_losses.dtype)
    loss_grid[sample_mask] = flat_losses
    counts = sample_mask.sum(dim=-1).float().clamp_min(1.0)
    return loss_grid.sum(dim=-1) / counts


class UILA(UPD):
    """
    Universal variant of Intermediate-Level Attack (ILA), adapted for LLMs. 
    - Enhancing Adversarial Example Transferability with an Intermediate Level Attack [https://arxiv.org/abs/1907.10823]
    """
    
    def __init__(self, *args, normalized_loss: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        self.normalized_loss = normalized_loss
        self.metric_tracker.report_hparams("attack", normalized_loss=self.normalized_loss)

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
                    METRICS["U-ILA/not_fooled_ratio"] = not_fooled_ratio

                    if fooled_mask.all():  # all samples already fooled
                        METRICS["U-ILA/effective_batch_ratio"] = 0.0
                        METRICS["loss"] = None
                        return METRICS

                    input_texts = [txt for txt, m in zip(input_texts, fooled_mask) if not m]
                    input_convs = [conv for conv, m in zip(input_convs, fooled_mask) if not m]
                    target_texts = [tgt for tgt, m in zip(target_texts, fooled_mask) if not m]

            # run per-sample attack
            with torch.autocast(device_type=self.device.type, enabled=False):
                init_embeds = None
                if position.epoch >= self.warmup_epochs:
                    # use universal embeds as init only after warmup
                    init_embeds = self.univ_embeds.repeat(len(input_convs), 1, 1)

                clean_convs = [[{"role": "user", "content": prm}] for prm in input_texts]
                sample_result = self.inner_attack.fit(clean_convs, target_texts, init_embeds=init_embeds)

                if sample_losses := sample_result.logs.get("loss"):
                    METRICS["U-ILA/initial_sample_attack_loss"] = sample_losses[0]

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
                    METRICS["U-ILA/sample_attack_success_ratio"] = sample_asr

                    if not success_mask.any():  # all attacks failed
                        METRICS["U-ILA/effective_batch_ratio"] = 0.0
                        METRICS["loss"] = None
                        return METRICS

                    if self.dynamic_labels > 0:
                        # set target texts to generated sample responses
                        target_texts = self.truncate_tokens(sample_responses, self.dynamic_labels)

                    input_convs = [conv for conv, m in zip(input_convs, success_mask) if m]
                    target_texts = [tgt for tgt, m in zip(target_texts, success_mask) if m]
                    clean_convs = [conv for conv, m in zip(clean_convs, success_mask) if m]
                    sample_result = sample_result.masked_select(success_mask)

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

            batch_ratio = len(input_convs) / len(data["prompt"])
            METRICS["U-ILA/effective_batch_ratio"] = batch_ratio

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

                # compute clean activations
                clean_encodings = self.adv_model.tokenize(
                    clean_convs,
                    target_texts,
                    target_controls=self.target_controls,
                )

                with torch.inference_mode():
                    self.adv_model.forward(
                        input_ids=clean_encodings.input_ids,
                        attention_mask=clean_encodings.attention_mask,
                        adv_mask=None,
                    )
                    clean_activs = self.activ_extractor.get_activations()

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
            
            criterion = ActivationLoss(loss_fn=ila_loss, reduction="sum-mean")
            loss = criterion.forward(
                univ_activs,
                sample_activs,
                clean_activs,
                univ_mask=univ_encodings.target_mask,
                sample_mask=sample_encodings.target_mask,
                clean_mask=clean_encodings.target_mask,
                sample_mean=False,
                normalized=self.normalized_loss,
            )
            

        # grad step
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        METRICS["loss"] = loss.item()
        return METRICS
