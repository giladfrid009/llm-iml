from src.attack import Attack
from src.adver_model import AdverModel
from src.data import DF_Batcher
from src.eval.evaluator import Evaluator

from typing import Any
import torch
from tqdm.auto import tqdm
import time
from typing import Iterable, Callable
import warnings


class StopCriteria:
    def __init__(
        self,
        max_epochs: int = 100,
        max_evals: int | None = None,
        max_time: float | None = None,
        target_value: float | None = None,
        patience: int | None = None,
        patience_delta: float = 1e-4,
    ):
        """
        Container for various stopping criteria.

        Args:
            max_epochs: Maximum number of epochs.
            max_evals: Maximum number of evaluation steps.
            max_time: Maximum training time in seconds.
            target_value: Target value to stop training when reached. This value should be maximized.
            patience: Number of evaluation steps without sufficient improvement.
            patience_delta: Minimum improvement delta to reset patience.

        Raises:
            ValueError: If any of the arguments are invalid.
        """
        if max_epochs <= 0:
            raise ValueError("Max epochs must be greater than 0.")
        if max_evals is not None and max_evals <= 0:
            raise ValueError("Max evals must be greater than 0.")
        if max_time is not None and max_time <= 0:
            raise ValueError("Max time must be greater than 0.")
        if patience is not None and patience <= 0:
            raise ValueError("Patience must be greater than 0.")
        if patience_delta < 0:
            raise ValueError("Patience delta must be greater than or equal to 0.")

        self.max_epochs = max_epochs
        self.max_evals = max_evals
        self.max_time = max_time
        self.target_value = target_value
        self.patience = patience
        self.patience_delta = patience_delta

        # Internal state
        self._epoch = 0
        self._total_evals = 0
        self._start_time = time.time()
        self._best_value = -float("inf")
        self._patience_counter = 0
        self.reset()

    def get_hparams(self) -> dict:
        return {
            "stop/max_epochs": self.max_epochs,
            "stop/max_evals": self.max_evals,
            "stop/max_time": self.max_time,
            "stop/target_value": self.target_value,
            "stop/patience": self.patience,
            "stop/patience_delta": self.patience_delta,
        }

    def reset(self) -> None:
        """Reset internal state."""
        self._epoch = 0
        self._total_evals = 0
        self._start_time = time.time()
        self._best_value = -float("inf")
        self._patience_counter = 0

    def update(self, epoch: int, value: float | None) -> None:
        """Update internal state with new metrics."""
        self._epoch = epoch
        self._total_evals += 1

        if value is not None:
            if (value - self._best_value) >= self.patience_delta:
                self._best_value = value
                self._patience_counter = 0
            else:
                self._patience_counter += 1

    def should_stop(self) -> bool:
        """Check if any stopping condition is met."""
        if self.target_value is not None and self._best_value >= self.target_value:
            print(f"Stopping: Target value reached :: ({self.target_value})")
            return True

        if self._epoch >= self.max_epochs:
            print(f"Stopping: Max epochs reached :: ({self.max_epochs})")
            return True

        if self.max_evals is not None and self._total_evals >= self.max_evals:
            print(f"Stopping: Max evals reached :: ({self.max_evals})")
            return True

        if self.patience is not None and self._patience_counter >= self.patience:
            print(f"Stopping: Patience exceeded :: ({self.patience})")
            return True

        if self.max_time is not None and (time.time() - self._start_time) > self.max_time:
            print(f"Stopping: Max time reached :: ({self.max_time} sec)")
            return True

        return False


class IML_Attack:
    def __init__(
        self,
        adv_model: AdverModel,
        internal_attack: Attack,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        mixed_precision: bool = True,
        evaluators: list[Evaluator] | None = None,
        pred_kwargs: dict[str, Any] | None = None,
    ):
        self.adv_model = adv_model
        self.internal_attack = internal_attack

        self.evaluators = evaluators
        self.pred_kwargs = pred_kwargs if pred_kwargs is not None else {}

        self.mixed_precision = mixed_precision
        self.grad_scaler = torch.GradScaler(enabled=mixed_precision)

        self.univ_embeds = self.init_embedding()
        self.optimizer = optim_factory([self.univ_embeds])

        self.best_metric = -float("inf")
        self.best_embeds = self.univ_embeds.clone().detach()

    @property
    def num_tokens(self) -> int:
        return self.internal_attack.num_tokens

    @property
    def embed_dim(self) -> int:
        return self.internal_attack.embed_dim

    @property
    def device(self) -> torch.device:
        return self.internal_attack.device

    def init_embedding(self) -> torch.Tensor:
        return torch.randn(
            size=(1, self.num_tokens, self.embed_dim),
            device=self.device,
            dtype=self.internal_attack.embed_dtype,
            requires_grad=True,
        )

    def get_univ_embed(self) -> torch.Tensor:
        return self.univ_embeds.clone().detach()

    @torch.inference_mode()
    def predict(
        self,
        adv_model: AdverModel,
        dl_eval: DF_Batcher,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generates model responses for the evaluation data loader.

        Args:
            adv_model (AdverModel): The adversarial model to use for text generation.
            dl_eval (DF_Batcher): Data loader for evaluation.
            **kwargs (dict): Additional keyword arguments for `AdverModel.generate_text`.

        Returns:
            list[str]: List of generated responses.
        """

        kwargs = kwargs if kwargs else self.pred_kwargs
        dl_eval = dl_eval.copy(shuffle=False, drop_last=False)

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):

            all_responses = []
            for batch_data in tqdm(dl_eval, desc="Predict", leave=False):
                prompts = batch_data.prompt
                system_text = batch_data.system if hasattr(batch_data, "system") else None

                responses = adv_model.generate_text(
                    prompts,
                    system_text=system_text,
                    **kwargs,
                )

                all_responses.extend(responses)

        return all_responses

    @torch.inference_mode()
    def evaluate(
        self,
        adv_model: AdverModel,
        evalers: list[Evaluator],
        dl_eval: DF_Batcher,
        update_best: bool = False,
    ) -> dict[str, float]:
        """
        Evaluates the model using the provided evaluators and data loader.

        Args:
            adv_model (AdverModel): The adversarial model to evaluate.
            evalers (list[Evaluator]): List of evaluators to use for evaluation.
            dl_eval (DF_Batcher): Data loader for evaluation.
            update_embeds (bool): If True, updates the universal embeddings with the best ones found during evaluation.

        Returns:
            dict[str, float]: Dictionary containing evaluation metrics.
        """
        dl_eval = dl_eval.copy(shuffle=False, drop_last=False)

        all_responses = self.predict(adv_model, dl_eval)
        dl_eval.set_column("response", all_responses)

        metrics = {}
        for evaluator in evalers:
            metrics[evaluator.name] = evaluator.evaluate(dl_eval)

        if update_best:
            if self.best_metric < metrics[evalers[0].name]:
                self.best_metric = metrics[evalers[0].name]
                self.best_embeds = self.adv_model.get_embeddings(clone=True)

        return metrics

    # TODO: CLEAN UP AND SPLIT INTO SMALLER FUNTIONS
    # ITS BECOMING A MESS AND NON-MAINTANABLE, only after AdverModel refactor

    # TODO: probably we need to make the evaluators mandatory
    # in order for IML to only use succesful attacks

    # TODO: after this point i feel we can practically use the same code as we use in ulib
    # including all the logging, etc...
    def fit(
        self,
        dl_train: DF_Batcher,
        dl_eval: DF_Batcher | None = None,
        stop_criteria: StopCriteria | None = None,
    ) -> AdverModel:

        if dl_eval is not None and self.evaluators is None:
            warnings.warn("Evaluation data loader provided but no evaluators specified. Skipping evaluation.")

        # must have these columns at least
        dl_train.validate(["prompt", "target"])

        if stop_criteria is None:
            stop_criteria = StopCriteria()

        loss_value = None
        stop_criteria.reset()
        should_stop = stop_criteria.should_stop()

        with tqdm(range(stop_criteria.max_epochs), desc="Epochs") as epoch_pbar:

            # initial evaluation
            if dl_eval is not None and self.evaluators is not None:
                self.adv_model.set_embeddings(self.univ_embeds)
                metrics = self.evaluate(
                    adv_model=self.adv_model,
                    evalers=self.evaluators[:1],
                    dl_eval=dl_eval,
                    update_best=True,
                )

                stop_criteria.update(0, metrics[self.evaluators[0].name])
                epoch_pbar.set_postfix(metrics)

            # main training loop
            for epoch_num in epoch_pbar:
                if should_stop:
                    break

                with tqdm(dl_train, desc="Batch", leave=False) as batch_pbar:
                    for batch_num, batch_data in enumerate(batch_pbar):
                        if should_stop:
                            break

                        loss_value = self.optim_step(batch_data, epoch_num, batch_num)

                        stop_criteria.update(epoch_num, None)
                        batch_pbar.set_postfix({"loss": loss_value})

                # evaluate after each epoch
                if dl_eval is not None and self.evaluators is not None:
                    self.adv_model.set_embeddings(self.univ_embeds)
                    metrics = self.evaluate(
                        adv_model=self.adv_model,
                        evalers=self.evaluators,
                        dl_eval=dl_eval,
                        update_best=True,
                    )

                    stop_criteria.update(epoch_num, metrics[self.evaluators[0].name])
                    epoch_pbar.set_postfix(metrics)

                else:
                    stop_criteria.update(epoch_num, None)
                    epoch_pbar.set_postfix({"loss": loss_value})

        # set to best
        self.adv_model.set_embeddings(self.best_embeds)

        # final evaluation
        if dl_eval is not None and self.evaluators is not None:
            metrics = self.evaluate(self.adv_model, self.evaluators, dl_eval)
            for name, value in metrics.items():
                print(f"Final metric {name}: {value:.6f}")

        return self.adv_model

    def optim_step(self, data: tuple[list[Any], ...], epoch_num: int, batch_num: int) -> float | None:
        """
        Perform a single optimization step on the given batch of data.

        Args:
            data (tuple[list[Any], ...]): Batch data containing input and target text
                via attributes `data.prompt` and `data.target`.
            epoch_num (int): Current epoch number.
            batch_num (int): Current batch number.

        Returns:
            float | None: Loss value for the optimization step, or None if no loss is computed.

        """
        self.optimizer.zero_grad()

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):

            input_text, target_text = data.prompt, data.target
            token_dict = self.adv_model.tokenize(input_text, target_text)

            # compute universal logits
            univ_embeds = self.univ_embeds.expand(len(input_text), -1, -1)
            self.adv_model.set_embeddings(univ_embeds)
            univ_logits = self.compute_logits(token_dict, self.adv_model)

            # compute per-sample logits
            with torch.autocast(device_type=self.device.type, enabled=False):
                sample_embed = self.internal_attack.fit(input_text, target_text, embeds_init=univ_embeds)
            with torch.inference_mode():
                self.adv_model.set_embeddings(sample_embed)
                sample_logits = self.compute_logits(token_dict, self.adv_model)

            # compute loss
            univ_logits = univ_logits.view(-1, univ_logits.size(-1))
            sample_logits = sample_logits.view(-1, sample_logits.size(-1))
            loss = 1 - torch.cosine_similarity(univ_logits, sample_logits, dim=-1).mean()

        if loss is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

        return loss.item() if loss is not None else None

    def compute_logits(self, token_dict: dict, adv_model: AdverModel) -> torch.Tensor:

        result = adv_model.forward(
            input_ids=token_dict["input_ids"],
            attention_mask=token_dict["attention_mask"],
            adv_mask=token_dict["adv_mask"],
        )

        logits, _ = self.internal_attack.align_preds(
            result.logits,
            input_ids=token_dict["input_ids"],
            target_mask=token_dict["target_mask"],
        )

        return logits
