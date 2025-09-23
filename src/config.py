from transformers.generation.configuration_utils import GenerationConfig
from typing import Any
import time
from src.utils.logging import create_logger

logger = create_logger(__name__)


class GenConfig:
    """
    A configuration class for generation parameters, similar to `GenerationConfig` from Hugging Face Transformers.

    Important:
    When generating, for all parameters marked as `None` the model default generation config value will be used.
    """

    def __init__(
        self,
        *,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        max_time: float | None = None,
        **kwargs,
    ):
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_time = max_time

        for key, value in kwargs.items():
            setattr(self, key, value)

        # to fix annoying do_sample warnings:
        if self.do_sample is False:
            self.temperature = 1.0
            self.top_p = 1.0
            self.min_p = None
            self.typical_p = 1.0
            self.top_k = 50
            self.epsilon_cutoff = 0.0
            self.eta_cutoff = 0.0

    def get_hparams(self) -> dict[str, Any]:
        """
        Returns a dictionary of all non-None parameters in this config.
        """
        params = {}
        for name, value in vars(self).items():
            if not name.startswith("_") and value is not None:
                params[name] = value
        return params

    def patch_params(
        self,
        generation_config: GenerationConfig | None = None,
        param_dict: dict[str, Any] | None = None,
    ):
        """
        Update any parameter in the provided arguments with non-None values from this config.

        Args:
            generation_config (GenerationConfig | None): The GenerationConfig object to update.
            param_dict (dict[str, Any] | None): A dictionary of parameters to update.
        """

        params = self.get_hparams()

        if generation_config is not None:
            for name, value in params.items():
                if hasattr(generation_config, name):
                    setattr(generation_config, name, value)
                else:
                    raise ValueError(f"Field '{name}' not found in GenerationConfig.")

        if param_dict is not None:
            for name, value in params.items():
                if name in param_dict:
                    param_dict[name] = value

    def update(self, **kwargs) -> dict[str, Any]:
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`): Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


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
            logger.info(f"Stopping: Target value reached :: ({self.target_value})")
            return True

        if self._epoch >= self.max_epochs:
            logger.info(f"Stopping: Max epochs reached :: ({self.max_epochs})")
            return True

        if self.max_evals is not None and self._total_evals >= self.max_evals:
            logger.info(f"Stopping: Max evals reached :: ({self.max_evals})")
            return True

        if self.patience is not None and self._patience_counter >= self.patience:
            logger.info(f"Stopping: Patience exceeded :: ({self.patience})")
            return True

        if self.max_time is not None and (time.time() - self._start_time) > self.max_time:
            logger.info(f"Stopping: Max time reached :: ({self.max_time} sec)")
            return True

        return False
