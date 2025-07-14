from src.adv_model import AdvModel
from src.data import DF_Batcher
from src.eval.evaluator import Evaluator
from src.logger import Logger
from src.config import GenConfig, StopCriteria

from typing import Any
import torch
from tqdm.auto import tqdm
import warnings
import pathlib
from abc import abstractmethod


class UnivAttack:
    def __init__(
        self,
        adv_model: AdvModel,
        evaluators: list[Evaluator],
        eval_freq: int | float = 1,
        mixed_precision: bool = True,
        gen_config: GenConfig | None = None,
        log_dir: str | None = None,
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
            log_dir (str | None): Directory to save logs. If None, no logging is performed.
        """
        self.adv_model = adv_model
        self.evaluators = evaluators
        self.eval_freq = eval_freq

        if gen_config is None:
            gen_config = GenConfig()

        self.gen_config = gen_config

        self.mixed_precision = mixed_precision
        self.grad_scaler = torch.GradScaler(enabled=mixed_precision)

        self.univ_embeds = self.adv_model.get_embeddings(clone=False)
        self.univ_embeds.requires_grad_(True)
        if self.univ_embeds.size(0) != 1:
            raise ValueError("Batch size of universal embeddings must be 1.")

        self.best_metric = -float("inf")
        self.best_embeds = self.univ_embeds.clone().detach()

        # logging
        self.logger = Logger(log_dir)
        self.logger.register_hparams(self.get_hparams())
        self.logger.register_hparams(adv_model.get_hparams())
        for ev in self.evaluators:
            self.logger.register_hparams(ev.get_hparams())
        self.logger.register_hparams({f"grad_scaler/{k}": v for k, v in self.grad_scaler.state_dict().items()})

    @property
    def num_tokens(self) -> int:
        return self.adv_model.num_tokens

    @property
    def embed_dim(self) -> int:
        return self.adv_model.adv_embedder.embed_dim

    @property
    def device(self) -> torch.device:
        return self.adv_model.device

    @property
    def judge(self) -> Evaluator:
        """
        Returns the main judge evaluator.
        This is the first evaluator in the list of evaluators.
        """
        return self.evaluators[0]

    def get_hparams(self) -> dict:
        return {
            "univ_attack/num_tokens": self.num_tokens,
            "univ_attack/embed_dim": self.embed_dim,
            "univ_attack/mixed_precision": self.mixed_precision,
            "univ_attack/eval_freq": self.eval_freq,
            "univ_attack/gen_config": self.gen_config.get_hparams() if self.gen_config else {},
            "univ_attack/evaluators": (e.name for e in self.evaluators),
            "univ_attack/log_dir": self.logger.root_dir if self.logger else None,
        }

    def close(self):
        """Close all resources."""
        self.logger.close()

    def __del__(self):
        self.close()

    def save_checkpoint(self, file_name: str = "best_embeds.pt"):
        log_dir = self.logger.log_dir()
        if log_dir is not None:
            torch.save(self.best_embeds, pathlib.Path(log_dir) / file_name)

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
    ) -> list[float]:
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
            list[float]: List containing evaluation metrics from each evaluator.
        """

        if isinstance(evaluators, Evaluator):
            evaluators = [evaluators]

        # TODO: (low priority) add support to evaluating multiple generations per prompt
        # easiest and probably cleanest solution is to copy each row in dl_eval multiple times

        if dl_eval.drop_last or dl_eval.shuffle:
            warnings.warn(
                "Evaluation data loader should not be shuffled or dropped last. "
                "Creating a shallow copy with `shuffle=False` and `drop_last=False`."
            )
            dl_eval = dl_eval.copy(shuffle=False, drop_last=False)

        self.predict(adv_model=adv_model, dl=dl_eval, config=gen_config, **kwargs)

        metrics = []
        for evaluator in evaluators:
            res = evaluator.evaluate(dl_eval)
            metrics.append(res)

        if update_best:
            # Find metric which belongs to the judge evaluator
            # If it improves, update the best metric and embeddings
            judge_metric = None
            for ev, m in zip(evaluators, metrics):
                if ev == self.judge:
                    judge_metric = m
                    break

            if judge_metric is None:
                raise ValueError(
                    f"Judge evaluator {self.judge.name} not found in the list of evaluators {[e.name for e in evaluators]}."
                )

            if self.best_metric < judge_metric:
                self.best_metric = judge_metric
                self.best_embeds = adv_model.get_embeddings(clone=True)

        return metrics

    def fit(
        self,
        dl_train: DF_Batcher,
        dl_eval: DF_Batcher | None = None,
        stop_criteria: StopCriteria | None = None,
    ) -> AdvModel:

        if dl_eval is None:
            warnings.warn("No evaluation data loader provided, using training data for evaluation.")
            dl_eval = dl_train.copy(shuffle=False, drop_last=False)

        # must have these columns at least
        dl_train.validate(["prompt", "target"])

        if stop_criteria is None:
            stop_criteria = StopCriteria()

        # local stats
        global_step = 0
        loss_value = None
        stop_criteria.reset()
        should_stop = stop_criteria.should_stop()

        # init logging
        self.logger.register_hparams(stop_criteria.get_hparams())

        self.logger.initialize(
            self.adv_model.model.name_or_path,
            f"num_tokens_{self.num_tokens}",
            self.__class__.__name__,
        )
        self.logger.add_tags(
            model=self.adv_model.model.name_or_path,
            num_tokens=self.num_tokens,
            attack=self.__class__.__name__,
        )
        self.logger.log_hparams()

        with tqdm(range(stop_criteria.max_epochs), desc="Epochs") as epoch_pbar:

            # initial evaluation
            self.adv_model.set_embeddings(self.univ_embeds)
            metrics = self.evaluate(self.adv_model, self.evaluators, dl_eval, update_best=True)

            self.save_checkpoint()
            self.logger.log_scalar(f"{self.judge.name}/best", self.best_metric, step=-1)
            self.logger.log_scalers({f"{e.name}/current": m for e, m in zip(self.evaluators, metrics)}, step=-1)
            epoch_pbar.set_postfix({e.name: m for e, m in zip(self.evaluators, metrics)})

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
                        self.logger.log_scalar("loss", loss_value, step=global_step)
                        batch_pbar.set_postfix({"loss": loss_value})
                        should_stop = stop_criteria.should_stop()

                        # evaluation step
                        if should_stop or (
                            global_step > 0 and global_step % round(self.eval_freq * len(dl_train)) == 0
                        ):
                            self.adv_model.set_embeddings(self.univ_embeds)
                            metrics = self.evaluate(self.adv_model, self.evaluators, dl_eval, update_best=True)
                            stop_criteria.update(epoch_num, metrics[0])

                            self.save_checkpoint()
                            self.logger.log_scalar(f"{self.judge.name}/best", self.best_metric, step=global_step)
                            self.logger.log_scalers(
                                {f"{e.name}/current": m for e, m in zip(self.evaluators, metrics)}, step=global_step
                            )
                            epoch_pbar.set_postfix({e.name: m for e, m in zip(self.evaluators, metrics)})

                        global_step += 1

        # set to best embeddings and final eval
        self.adv_model.set_embeddings(self.best_embeds)
        metrics = self.evaluate(self.adv_model, self.evaluators, dl_eval)
        self.logger.log_metrics({e.name: m for e, m in zip(self.evaluators, metrics)})
        for ev, val in zip(self.evaluators, metrics):
            print(f"Final metric {ev.name}: {val:.6f}")

        self.close()
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
