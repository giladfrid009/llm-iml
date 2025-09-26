from src.adv_model import AdvModel
from src.data import DF_Batcher
from src.eval.evaluator import Evaluator
from src.metric_logger import MetricLogger
from src.config import GenConfig, StopCriteria
from src.utils.logging import create_logger

import time
from typing import Any
from tqdm.auto import tqdm
import pathlib
from abc import abstractmethod
import torch
import torch.nn.functional as F

logger = create_logger(__name__)


class UnivAttack:
    def __init__(
        self,
        adv_model: AdvModel,
        evaluators: list[Evaluator],
        judge_metric: str | None = None,
        eval_freq: int | float = 1,
        mixed_precision: bool = True,
        gen_config: GenConfig | None = None,
        log_dir: str = "logs",
    ):
        """
        Args:
            adv_model (AdvModel): The adversarial model to use for text generation.
            evaluators (list[Evaluator]): List of evaluators to use for evaluation.
            eval_freq (int | float): Frequency of evaluation during training.
                - if int, evaluates every `eval_freq` epochs.
                - if float, evaluates every `round(eval_freq * len(dl_train))` batches.
            mixed_precision (bool): Whether to use mixed precision training.
            gen_config (GenConfig  | None): Default generation configuration.
            log_dir (str): Directory to save logs.
        """
        if gen_config is None:
            gen_config = GenConfig()

        if judge_metric is None:
            judge_metric = evaluators[0].metric_names[0]

        self.adv_model = adv_model
        self.evaluators = evaluators
        self.judge_metric = judge_metric
        self.eval_freq = eval_freq
        self.gen_config = gen_config
        self.mixed_precision = mixed_precision
        self.grad_scaler = torch.GradScaler(enabled=mixed_precision)

        # prepare for optimization
        self.univ_embeds.requires_grad_(True)
        if self.univ_embeds.size(0) != 1:
            raise ValueError("Batch size of universal embeddings must be 1.")

        # local params
        self.best_metric = -float("inf")
        self.best_embeds = self.univ_embeds.clone().detach()

        # logging
        self.metric_logger = MetricLogger(
            time.strftime("%Y-%m-%d_%H-%M-%S"),
            project="LLM-IML",
            root_dir=log_dir,
        )

        self.metric_logger.add_tags(
            model=self.adv_model.model.name_or_path,
            num_tokens=self.num_tokens,
            attack=self.__class__.__name__,
        )

        self.metric_logger.log_hparams(
            "univ_attack",
            num_tokens=self.num_tokens,
            mixed_precision=self.mixed_precision,
            eval_freq=self.eval_freq,
            gen_config=self.gen_config.get_hparams(),
            evaluators=[e.name for e in self.evaluators],
            judge_metric=self.judge_metric,
            log_dir=self.metric_logger.root_dir if self.metric_logger else None,
        )

        self.metric_logger.log_hparams("adv_model", adv_model.get_hparams())
        self.metric_logger.log_hparams("grad_scaler", self.grad_scaler.state_dict())
        for ev in self.evaluators:
            self.metric_logger.log_hparams(f"evaluators/{ev.name}", ev.get_hparams())

    @property
    def univ_embeds(self) -> torch.Tensor:
        return self.adv_model.get_embeddings(clone=False)

    @property
    def num_tokens(self) -> int:
        return self.adv_model.num_tokens

    @property
    def embed_dim(self) -> int:
        return self.adv_model.adv_embedder.embed_dim

    @property
    def device(self) -> torch.device:
        return self.adv_model.device

    def close(self):
        self.metric_logger.close()

    def save_checkpoint(self, file_name: str = "best_embeds.pt"):
        torch.save(self.best_embeds, pathlib.Path(self.metric_logger.log_dir) / file_name)

    @torch.inference_mode()
    def predict(
        self,
        adv_model: AdvModel,
        dl: DF_Batcher,
        config: GenConfig | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generates model responses for the evaluation data loader.
        Sets the "response" column in the data loader with the generated responses.

        Args:
            adv_model (AdvModel): The adversarial model to use for text generation.
            dl (DF_Batcher): Data loader with prompts for generation.
            config (GenConfig | None): Generation configuration to use for the model.
            **kwargs (dict): Additional keyword arguments for `AdvModel.chat`.

        Returns:
            list[str]: List of generated responses.
        """

        config = config if config else self.gen_config
        dl = dl.copy(shuffle=False, drop_last=False)

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            all_responses = []
            for batch_data in tqdm(dl, desc="Generating", leave=False):
                prompts = batch_data["prompt"]
                conversations = [[{"role": "user", "content": prm}] for prm in prompts]
                responses = adv_model.chat(conversations, config=config, **kwargs)
                all_responses.extend(responses)

        dl.set_column("response", all_responses)

        return all_responses

    @torch.inference_mode()
    def evaluate(
        self,
        adv_model: AdvModel,
        evaluators: list[Evaluator] | Evaluator,
        dl_eval: DF_Batcher,
        gen_config: GenConfig | None = None,
        update_best: bool = False,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Evaluates the model using the provided evaluators and data loader.

        Args:
            adv_model (AdvModel): The adversarial model to evaluate.
            evaluators (list[Evaluator]): List of evaluators to use for evaluation.
            dl_eval (DF_Batcher): Data loader for evaluation.
            gen_config (GenConfig | None): Generation configuration to use for the model.
            update_best (bool): If True, updates the best metric and embeddings if the current evaluation is better.
            **kwargs (dict): Additional keyword arguments for `AdvModel.chat`.

        Returns:
            dict[str, float]:
        """

        if isinstance(evaluators, Evaluator):
            evaluators = [evaluators]

        # TODO: (low priority) add support to evaluating multiple generations per prompt
        # easiest and probably cleanest solution is to copy each row in dl_eval multiple times

        if dl_eval.drop_last or dl_eval.shuffle:
            raise ValueError("dl_eval must have shuffle=False and drop_last=False")

        self.predict(adv_model=adv_model, dl=dl_eval, config=gen_config, **kwargs)

        all_metrics = {}
        for evaluator in evaluators:
            metrics = evaluator.evaluate(dl_eval)
            all_metrics.update(metrics)

        if update_best:
            judge_metric = all_metrics.get(self.judge_metric)
            if judge_metric is None:
                raise ValueError(
                    f"Judge metric `{self.judge_metric}` not found in the list of produced metrics {list(all_metrics.keys())}."
                )

            if self.best_metric < judge_metric:
                self.best_metric = judge_metric
                self.best_embeds = adv_model.get_embeddings(clone=True)

        return all_metrics

    # TODO: probably remove
    def compute_metrics(self) -> dict[str, float]:
        """Compute diagnostic metrics for the current universal embeddings.

        Namespaces (for clean logger grouping):
          univ.closest_token/*  : Stats wrt each token's closest vocab entry (cosine / L2 criteria)
          univ.vocab/*          : Aggregated stats over similarities to the entire vocab
          univ.mean_vocab/*     : Distance / similarity to the (unweighted) mean vocab embedding
          univ.proj/*           : Distances/similarities to the specific closest vocab tokens (by cosine & L2)
          univ.intra/*          : Diversity among the optimized universal tokens themselves

        All metrics are averaged across the optimized token dimension (N) when applicable.
        Only cosine similarities and L2 distances are reported (all L1 related code removed by request).
        """
        metrics: dict[str, float] = {}

        # Retrieve (1, N, D) universal embeddings; batch dimension must be 1 for a universal attack.
        univ = self.adv_model.get_embeddings(clone=True)
        if univ.size(0) != 1:
            raise ValueError("Universal embeddings batch size must be 1.")
        embeds = univ[0]  # (N, D)
        N, D = embeds.shape
        if N == 0:
            return metrics  # nothing to report

        # ------------------ Vocabulary preparation ------------------
        vocab: torch.Tensor = self.adv_model.orig_embedder.weight.detach().to(embeds.device)  # (V, D)
        V = vocab.size(0)
        vocab_mean = vocab.mean(dim=0)  # (D,)

        # Normalized versions for cosine similarity (avoid repeated division).
        embeds_norm = F.normalize(embeds, p=2, dim=-1)  # (N, D)
        vocab_norm = F.normalize(vocab, p=2, dim=-1)  # (V, D)
        vocab_mean_norm = F.normalize(vocab_mean, p=2, dim=0)  # (D,)

        # ------------------ Cosine similarity vs full vocab ------------------
        # Shape: (N, V)
        cos_sim = embeds_norm @ vocab_norm.T
        cos_max_vals, cos_max_idx = cos_sim.max(dim=-1)  # (N,)

        metrics["univ.closest_token/cos_max_mean"] = cos_max_vals.mean().item()
        metrics["univ.vocab/cos_mean_all"] = cos_sim.mean().item()
        k = min(5, V)
        metrics["univ.vocab/cos_top5_mean"] = cos_sim.topk(k, dim=-1).values.mean().item()
        for thr in (0.7, 0.8, 0.9):
            metrics[f"univ.closest_token/frac_cos_gt_{thr}"] = (cos_max_vals > thr).float().mean().item()

        # ------------------ L2 distance to vocab (vectorized) ------------------
        # Using: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b (stable & fast)
        embeds_sq = (embeds * embeds).sum(dim=-1, keepdim=True)  # (N, 1)
        vocab_sq = (vocab * vocab).sum(dim=-1).unsqueeze(0)  # (1, V)
        l2_sq = (embeds_sq + vocab_sq - 2 * (embeds @ vocab.T)).clamp_min(0)  # (N, V)
        l2_min_sq, l2_min_idx = l2_sq.min(dim=-1)  # (N,)
        l2_min = l2_min_sq.sqrt()
        metrics["univ.closest_token/l2_min_mean"] = l2_min.mean().item()
        metrics["univ.closest_token/l2_min_median"] = l2_min.median().item()

        # ------------------ Distances / similarity to mean vocab embedding ------------------
        diff_to_vocab_mean = embeds - vocab_mean
        mean_l2 = diff_to_vocab_mean.norm(p=2, dim=-1)
        mean_cos = (embeds_norm * vocab_mean_norm).sum(dim=-1)
        metrics["univ.mean_vocab/l2_mean"] = mean_l2.mean().item()
        metrics["univ.mean_vocab/cos_mean"] = mean_cos.mean().item()

        # ------------------ Projections vs the closest tokens (two criteria) ------------------
        vocab_closest_cos = vocab[cos_max_idx]  # (N, D)
        vocab_closest_l2 = vocab[l2_min_idx]  # (N, D)
        metrics["univ.proj/cosToken_cos_mean"] = F.cosine_similarity(embeds, vocab_closest_cos, dim=-1).mean().item()
        metrics["univ.proj/l2Token_cos_mean"] = F.cosine_similarity(embeds, vocab_closest_l2, dim=-1).mean().item()
        metrics["univ.proj/cosToken_l2_mean"] = (embeds - vocab_closest_cos).norm(p=2, dim=-1).mean().item()
        metrics["univ.proj/l2Token_l2_mean"] = (embeds - vocab_closest_l2).norm(p=2, dim=-1).mean().item()

        # ------------------ Intra-set diversity (pairwise among optimized tokens) ------------------
        if N > 1:
            # Cosine diversity (exclude diagonal)
            intra_cos = embeds_norm @ embeds_norm.T  # (N, N)
            mask = ~torch.eye(N, dtype=torch.bool, device=embeds.device)
            metrics["univ.intra/cos_mean"] = intra_cos[mask].mean().item()

            # L2 diversity
            embeds_sq_vec = embeds_sq.squeeze(-1)  # (N,)
            intra_l2_sq = (embeds_sq_vec.unsqueeze(1) + embeds_sq_vec.unsqueeze(0) - 2 * (embeds @ embeds.T)).clamp_min(
                0
            )
            intra_l2 = intra_l2_sq.sqrt()
            metrics["univ.intra/l2_mean"] = intra_l2[mask].mean().item()

            # Spread / stability diagnostics
            metrics["univ.intra/closest_cos_std"] = cos_max_vals.std(unbiased=False).item()
            metrics["univ.intra/dim_var_mean"] = embeds.var(dim=0, unbiased=False).mean().item()
        else:
            # Single token: no diversity; set to 0 for clarity.
            metrics["univ.intra/cos_mean"] = 0.0
            metrics["univ.intra/l2_mean"] = 0.0
            metrics["univ.intra/closest_cos_std"] = 0.0
            metrics["univ.intra/dim_var_mean"] = 0.0

        return metrics

    def fit(
        self,
        dl_train: DF_Batcher,
        dl_eval: DF_Batcher,
        stop_criteria: StopCriteria | None = None,
    ) -> AdvModel:
        # must have these columns at least
        dl_train.validate(["prompt", "target"])
        dl_eval.validate(["prompt"])

        if stop_criteria is None:
            stop_criteria = StopCriteria()

        # local stats
        step = 0
        loss_value = None
        stop_criteria.reset()
        should_stop = stop_criteria.should_stop()

        # log relevant stats
        self.metric_logger.log_hparams("stop", stop_criteria.get_hparams())
        self.metric_logger.log_hparams("data/train", dl_train.get_hparams())
        self.metric_logger.log_hparams("data/eval", dl_eval.get_hparams())

        with tqdm(range(stop_criteria.max_epochs), desc="Epochs") as epoch_pbar:
            # initial evaluation
            metrics = self.evaluate(self.adv_model, self.evaluators, dl_eval, update_best=True)

            self.save_checkpoint()
            self.metric_logger.report_scalar(f"{self.judge_metric} (best)", self.best_metric, step=-1)
            self.metric_logger.report_scalars(metrics, step=-1)
            epoch_pbar.set_postfix(metrics)

            # main training loop
            for epoch_num in epoch_pbar:
                if should_stop:
                    break

                with tqdm(dl_train, desc="Batches", leave=False) as batch_pbar:
                    for batch_num, batch_data in enumerate(batch_pbar):
                        if should_stop:
                            break

                        # training step
                        loss_value = self.optim_step(batch_data, epoch_num, batch_num)
                        stop_criteria.update(epoch_num, None)
                        if loss_value is not None:
                            self.metric_logger.report_scalar("loss", loss_value, step)
                        batch_pbar.set_postfix({"loss": loss_value})
                        should_stop = stop_criteria.should_stop()

                        # evaluation step
                        if should_stop or (step > 0 and step % round(self.eval_freq * len(dl_train)) == 0):
                            metrics = self.evaluate(self.adv_model, self.evaluators, dl_eval, update_best=True)
                            stop_criteria.update(epoch_num, metrics[self.judge_metric])

                            self.save_checkpoint()
                            self.metric_logger.report_scalar(f"{self.judge_metric} (best)", self.best_metric, step)
                            self.metric_logger.report_scalars(metrics, step)
                            epoch_pbar.set_postfix(metrics)

                            pert_metrics = self.compute_metrics()
                            self.metric_logger.report_scalars(pert_metrics, step)

                        step += 1

        # set to best embeddings
        self.adv_model.set_embeddings(self.best_embeds)
        return self.adv_model

    @abstractmethod
    def optim_step(self, data: dict[str, list[Any]], epoch_num: int, batch_num: int) -> float | None:
        """
        Perform a single optimization step on the given batch of data.

        Args:
            data (dict[str, list[Any]]): Batch data containing input and target texts.
            epoch_num (int): Current epoch number.
            batch_num (int): Current batch number.

        Returns:
            float | None: Loss value for the optimization step, or None if no loss is computed.

        """
        raise NotImplementedError("Subclasses must implement the optim_step method.")
