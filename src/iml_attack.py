from src.attack import Attack
from src.adver_model import AdverModel

import torch
from tqdm.auto import tqdm
import time
from typing import Iterable, Callable


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

        if value is not None and (value - self._best_value) >= self.patience_delta:
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
    ):
        self.adv_model = adv_model
        self.internal_attack = internal_attack

        self.grad_scaler = torch.GradScaler(enabled=mixed_precision)
        self.autocast = torch.autocast(device_type=self.device.type, enabled=mixed_precision)

        self.univ_embeds = self.init_embedding()
        self.optimizer = optim_factory([self.univ_embeds])

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

    def autocast_context(self, enabled: bool = True) -> torch.autocast:
        """
        Returns an autocast context manager with the given enabled state.
        - If `enabled=True` then autocast will be enabled only if `self.autocast.is_enabled()` is True.
        - If `enabled=False` then autocast will be disabled.

        Args:
            enabled (bool): Whether to enable autocast.

        Returns:
            torch.autocast: Autocast context manager.
        """
        if enabled:
            return self.autocast
        else:
            return torch.autocast(device_type=self.autocast.device, enabled=False)

    def fit(
        self,
        dl_train: Iterable,
        dl_eval: Iterable | None = None,
        stop_criteria: StopCriteria | None = None,
    ) -> torch.Tensor:

        if dl_eval is None:
            dl_eval = dl_train

        if stop_criteria is None:
            stop_criteria = StopCriteria()

        stop_criteria.reset()
        should_stop = stop_criteria.should_stop()

        with tqdm(range(stop_criteria.max_epochs), desc="Epochs") as epoch_pbar:
            for epoch_num in epoch_pbar:
                if should_stop:
                    break

                with tqdm(dl_train, desc="Batch", leave=False) as batch_pbar:
                    for batch_num, batch_data in enumerate(batch_pbar):
                        if should_stop:
                            break

                        loss = self.process_batch(batch_data, batch_num, epoch_num)
                        stop_criteria.update(epoch_num, None)
                        batch_pbar.set_postfix({"loss": loss})

                stop_criteria.update(epoch_num, None)
                epoch_pbar.set_postfix({"loss": loss})

        return self.get_univ_embed()

    def process_batch(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> float | None:
        """
        Runs a single training step on the given batch of data.

        Args:
            data (tuple[torch.Tensor, ...]): Batch data, already moved to the device.
            batch_num (int): Current batch number.
            epoch_num (int): Current epoch number.

        Returns:
            float: Computed loss value for the batch, or None if invalid.
        """
        self.optimizer.zero_grad()

        with self.autocast_context():
            loss = self.compute_loss(data, batch_num, epoch_num)

        if loss is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

        return None if loss is None else loss.item()

    def compute_loss(self, data: tuple[torch.Tensor, ...], batch_num: int, epoch_num: int) -> torch.Tensor | None:
        """
        Computes the loss on the given batch of data.
        The computed loss must be a scalar tensor that allows gradients to be backpropagated.

        Args:
            data (tuple[torch.Tensor, ...]): Batch data, already moved to the device.
            batch_num (int): Current batch number.
            epoch_num (int): Current epoch number.

        Returns:
            torch.Tensor: Loss tensor, or None if invalid.
        """
        input_text, target_text = data
        token_dict = self.adv_model.tokenize(input_text, target_text)
        univ_embeds = torch.broadcast_to(self.univ_embeds, (len(input_text), *self.univ_embeds.shape[1:]))

        # locally disable autocast for per-sample attack
        with self.autocast_context(enabled=False):
            sample_embed = self.internal_attack.fit(input_text, target_text, adv_embeds=univ_embeds)

        # compute logits for per-sample and univ embedding
        univ_logits = self.compute_logits(token_dict, univ_embeds)
        with torch.inference_mode():
            sample_logits = self.compute_logits(token_dict, sample_embed)

        # compute loss
        univ_logits = univ_logits.view(-1, univ_logits.size(-1))
        sample_logits = sample_logits.view(-1, sample_logits.size(-1))
        loss = 1 - torch.cosine_similarity(univ_logits, sample_logits, dim=-1).mean()
        return loss

    def compute_logits(self, token_dict: dict, adv_embeds: torch.Tensor) -> torch.Tensor:

        result = self.adv_model.forward(
            input_ids=token_dict["input_ids"],
            attention_mask=token_dict["attention_mask"],
            adv_embeds=adv_embeds,
            adv_mask=token_dict["adv_mask"],
        )

        logits, _ = self.internal_attack.align_preds(
            result.logits,
            input_ids=token_dict["input_ids"],
            target_mask=token_dict["target_mask"],
        )

        return logits
