from src.adv_model import AdvModel
from src.data import TableLoader
from src.eval.evaluator import Evaluator
from src.utils.trackers import MetricTracker
from src.config import GenConfig, StopCriteria
from src.utils.logging import create_logger
from src.utils.torch import clear_memory

import time
from functools import total_ordering
from dataclasses import dataclass
from typing import Any
from tqdm.auto import tqdm
import pathlib
from abc import abstractmethod
import torch

logger = create_logger(__name__)


@total_ordering
@dataclass
class TrainPosition:
    epoch: int
    batch: int
    step: int

    def __lt__(self, other) -> bool:
        if not isinstance(other, TrainPosition):
            return NotImplemented
        # compare by (epoch, batch, step), python automatically handles lexicographic comparison
        return (self.epoch, self.batch, self.step) < (other.epoch, other.batch, other.step)


class UnivAttack:
    def __init__(
        self,
        adv_model: AdvModel,
        evaluators: list[Evaluator],
        eval_metric: str | None = None,
        eval_freq: int | float = 1,
        mixed_precision: bool = False,
        gen_config: GenConfig | None = None,
        metric_tracker: MetricTracker | None = None,
    ):
        """
        Args:
            adv_model (AdvModel): The adversarial model to use for text generation.
            evaluators (list[Evaluator]): List of evaluators to use for evaluation.
            eval_metric (str | None): The main evaluation metric, used for selecting the best model.
            eval_freq (int | float): Frequency of evaluation during training.
                - if int, evaluates every `eval_freq` epochs.
                - if float, evaluates every `round(eval_freq * len(dl_train))` batches.
            mixed_precision (bool): Whether to use mixed precision training.
            gen_config (GenConfig  | None): Default generation configuration.
            metric_tracker (BaseTracker | None): Metric logger for logging experiment data and metrics.
        """
        if eval_metric is None:
            eval_metric = evaluators[0].default_metric
            logger.info(f"Auto-selected main eval_metric: {eval_metric}")

        if not any(eval_metric in ev.metric_names for ev in evaluators):
            raise ValueError(
                f"eval_metric `{eval_metric}` not found in the list of evaluators metrics. "
                f"Available Metrics: {[ev.metric_names for ev in evaluators]}."
            )

        if gen_config is None:
            gen_config = GenConfig()

        self.adv_model = adv_model
        self.evaluators = evaluators
        self.eval_metric = eval_metric
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
        if metric_tracker is None:
            metric_tracker = MetricTracker.create(
                time.strftime("%Y-%m-%d_%H-%M-%S"),
                project="UnivAttack",
                root_dir="logs",
                kind="wandb",
            )

        self.metric_tracker = metric_tracker

        self.metric_tracker.report_hparams(
            "attack",
            model_name=adv_model.model.name_or_path,
            num_tokens=self.num_tokens,
            mixed_precision=self.mixed_precision,
            eval_freq=self.eval_freq,
            evaluators=[e.name for e in self.evaluators],
            eval_metric=self.eval_metric,
            log_dir=self.metric_tracker.root_dir,
        )

        self.metric_tracker.report_hparams(
            "hf_model",
            model_config=adv_model.model.config.to_dict(),
            generation_config=adv_model.model.generation_config.to_dict(),  # type: ignore
            name=adv_model.model.name_or_path,
        )

        self.metric_tracker.report_hparams("logger", self.metric_tracker.get_hparams())
        self.metric_tracker.report_hparams("adv_model", adv_model.get_hparams())
        self.metric_tracker.report_hparams("gen_config", self.gen_config.get_hparams())
        self.metric_tracker.report_hparams("grad_scaler", self.grad_scaler.state_dict())
        for ev in self.evaluators:
            self.metric_tracker.report_hparams(f"evaluators/{ev.name}", ev.get_hparams())

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

    def save_checkpoint(self, embeds: torch.Tensor | None = None, file_name: str = "best_embeds.pt"):
        if self.metric_tracker.log_dir is None:
            logger.warning("Log dir is None, cannot save checkpoint.")
            return

        if embeds is None:
            embeds = self.best_embeds

        torch.save(embeds, pathlib.Path(self.metric_tracker.log_dir) / file_name)

    @torch.inference_mode()
    def predict(
        self,
        dl: TableLoader,
        config: GenConfig | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generates model responses for the evaluation data loader.
        Sets the "response" column in the data loader with the generated responses.

        Args:
            dl (TableLoader): Data loader with prompts for generation.
            config (GenConfig | None): Generation configuration to use for the model.
            **kwargs (dict): Additional keyword arguments for `AdvModel.chat`.

        Returns:
            list[str]: List of generated responses.
        """
        if dl.drop_last or dl.shuffle:
            raise ValueError("dl must have shuffle=False and drop_last=False")

        config = config if config else self.gen_config

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            all_responses = []
            for batch_data in tqdm(dl, desc="Generating", leave=False):
                prompts = batch_data["prompt"]
                conversations = [[{"role": "user", "content": prm}] for prm in prompts]
                conversations = self.adv_model.inject_tokens(conversations)
                responses = self.adv_model.chat(conversations, self.univ_embeds, config, **kwargs)
                all_responses.extend(responses)

        dl.set_column("response", all_responses)

        return all_responses

    @torch.inference_mode()
    def evaluate(
        self,
        evaluators: list[Evaluator] | Evaluator,
        dl_eval: TableLoader,
        gen_config: GenConfig | None = None,
        update_best: bool = False,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Evaluates the model using the provided evaluators and data loader.

        Args:
            evaluators (list[Evaluator]): List of evaluators to use for evaluation.
            dl_eval (TableLoader): Data loader for evaluation.
            gen_config (GenConfig | None): Generation configuration to use for the model.
            update_best (bool): If True, updates the best metric and embeddings if the current evaluation is better.
            **kwargs (dict): Additional keyword arguments for `AdvModel.chat`.

        Returns:
            dict[str, float]:
        """
        if dl_eval.drop_last or dl_eval.shuffle:
            raise ValueError("dl_eval must have shuffle=False and drop_last=False")

        if isinstance(evaluators, Evaluator):
            evaluators = [evaluators]

        self.predict(dl_eval, config=gen_config, **kwargs)

        all_metrics: dict[str, float] = {}
        for evaluator in evaluators:
            metrics = evaluator.evaluate(dl_eval)
            all_metrics.update(metrics)

        if update_best:
            value = all_metrics.get(self.eval_metric)
            if value is None:
                raise ValueError(f"Eval metric `{self.eval_metric}` not found in the list of produced metrics {list(all_metrics.keys())}.")

            if self.best_metric < value:
                self.best_metric = value
                self.best_embeds = self.adv_model.get_embeddings(clone=True)

        return all_metrics

    def fit(
        self,
        dl_train: TableLoader,
        dl_eval: TableLoader,
        stop_criteria: StopCriteria | None = None,
    ) -> AdvModel:
        # must have these columns at least
        dl_train.validate(["prompt", "target"])
        dl_eval.validate(["prompt"])

        if stop_criteria is None:
            stop_criteria = StopCriteria()

        # local stats
        step = 0
        stop_criteria.reset()
        should_stop = stop_criteria.should_stop()

        # log relevant stats
        self.metric_tracker.report_hparams("stop", stop_criteria.get_hparams())
        self.metric_tracker.report_hparams("data/train", dl_train.get_hparams())
        self.metric_tracker.report_hparams("data/eval", dl_eval.get_hparams())

        with tqdm(range(stop_criteria.max_epochs), desc="Epochs") as epoch_pbar:
            # initial evaluation
            eval_metrics = self.evaluate(self.evaluators, dl_eval, update_best=True)
            clear_memory()

            self.save_checkpoint(self.best_embeds)
            self.metric_tracker.report_scalar(f"{self.eval_metric} (best)", self.best_metric, step=-1)
            self.metric_tracker.report_scalars(eval_metrics, step=-1)
            epoch_pbar.set_postfix(eval_metrics)

            # main training loop
            for epoch_num in epoch_pbar:
                if should_stop:
                    break

                with tqdm(dl_train, desc="Batches", leave=False) as batch_pbar:
                    for batch_num, batch_data in enumerate(batch_pbar):
                        if should_stop:
                            break

                        # training step
                        position = TrainPosition(epoch_num, batch_num, step)
                        batch_metrics = self.optim_step(batch_data, position)
                        stop_criteria.update(epoch_num, None)
                        self.metric_tracker.report_scalars(batch_metrics, step)
                        batch_pbar.set_postfix(batch_metrics)
                        should_stop = stop_criteria.should_stop()

                        # evaluation step
                        if should_stop or (step > 0 and step % round(self.eval_freq * len(dl_train)) == 0):
                            clear_memory()
                            eval_metrics = self.evaluate(self.evaluators, dl_eval, update_best=True)
                            stop_criteria.update(epoch_num, eval_metrics[self.eval_metric])
                            clear_memory()

                            self.save_checkpoint(self.best_embeds)
                            self.metric_tracker.report_scalar(f"{self.eval_metric} (best)", self.best_metric, step)
                            self.metric_tracker.report_scalars(eval_metrics, step)
                            epoch_pbar.set_postfix(eval_metrics)

                        step += 1

        # set to best embeddings
        self.adv_model.set_embeddings(self.best_embeds)
        return self.adv_model

    @abstractmethod
    def optim_step(self, data: dict[str, list[Any]], position: TrainPosition) -> dict[str, float | None]:
        """
        Perform a single optimization step on the given batch of data.

        Args:
            data (dict[str, list[Any]]): Batch data containing input and target texts.
            position (TrainPosition): Current position in training (epoch, batch, step).

        Returns:
            (dict[str, Any]): A dictionary containing relevant metrics from the optimization step.
        """
        raise NotImplementedError("Subclasses must implement the optim_step method.")
