from src.adv_model import AdvModel
from src.data import TableLoader
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
        metric_logger: MetricLogger | None = None,
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
            metric_logger (MetricLogger | None): Metric logger for logging experiment data and metrics.
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
        if metric_logger is None:
            metric_logger = MetricLogger(
                time.strftime("%Y-%m-%d_%H-%M-%S"),
                project="LLM-IML",
                root_dir="logs",
            )

        self.metric_logger = metric_logger

        self.metric_logger.log_hparams(
            "univ_attack",
            model_name=adv_model.model.name_or_path,
            num_tokens=self.num_tokens,
            mixed_precision=self.mixed_precision,
            eval_freq=self.eval_freq,
            evaluators=[e.name for e in self.evaluators],
            judge_metric=self.judge_metric,
            log_dir=self.metric_logger.root_dir,
        )

        self.metric_logger.log_hparams(
            "hf_model",
            model_config=adv_model.model.config.to_dict(),
            generation_config=adv_model.model.generation_config.to_dict(),  # type: ignore
            name=adv_model.model.name_or_path,
        )

        self.metric_logger.log_hparams("logger", self.metric_logger.get_hparams())
        self.metric_logger.log_hparams("adv_model", adv_model.get_hparams())
        self.metric_logger.log_hparams("gen_config", self.gen_config.get_hparams())
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
        dl: TableLoader,
        config: GenConfig | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generates model responses for the evaluation data loader.
        Sets the "response" column in the data loader with the generated responses.

        Args:
            adv_model (AdvModel): The adversarial model to use for text generation.
            dl (TableLoader): Data loader with prompts for generation.
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
        dl_eval: TableLoader,
        gen_config: GenConfig | None = None,
        update_best: bool = False,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Evaluates the model using the provided evaluators and data loader.

        Args:
            adv_model (AdvModel): The adversarial model to evaluate.
            evaluators (list[Evaluator]): List of evaluators to use for evaluation.
            dl_eval (TableLoader): Data loader for evaluation.
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
