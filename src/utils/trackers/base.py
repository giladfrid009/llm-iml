from __future__ import annotations
import pathlib
from typing import Any
import numpy as np
import PIL.Image
import torch
from abc import ABC, abstractmethod
from src.utils.logging import create_logger


logger = create_logger(__name__)


BACKEND_KINDS = {"wandb", "clearml"}


class MetricTracker(ABC):
    @staticmethod
    def create(*names: str, kind: str, project: str, root_dir: str = "logs", disabled: bool = False, **kwargs) -> MetricTracker:
        kind = kind.lower()

        if kind not in BACKEND_KINDS:
            raise ValueError(f"Unknown tracker kind: {kind}. Available kinds are {BACKEND_KINDS}.")

        from src.utils.trackers.wandb_backend import WandbTracker
        from src.utils.trackers.clearml_backend import ClearmlTracker

        if kind == "wandb":
            return WandbTracker(*names, project=project, root_dir=root_dir, disabled=disabled, **kwargs)
        elif kind == "clearml":
            return ClearmlTracker(*names, project=project, root_dir=root_dir, disabled=disabled, **kwargs)
        else:
            raise ValueError(f"Unknown tracker kind: {kind}")

    def __init__(
        self,
        *names: str,
        project: str,
        root_dir: str = "logs",
        disabled: bool = False,
    ):
        if len(names) == 0 and not disabled:
            raise ValueError("At least one name component must be provided.")

        self.run_name = str.join(" - ", names)
        self.project = project
        self.root_dir = root_dir
        self.disabled = disabled

        self.log_dir: str | None = None

        if not disabled:
            self.log_dir = self._create_directory(root_dir, project, *names)

    def get_hparams(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "project": self.project,
            "root_dir": self.root_dir,
            "log_dir": self.log_dir,
            "disabled": self.disabled,
        }

    def _create_directory(self, *subdir_parts: str) -> str:
        log_path = pathlib.Path(*subdir_parts)
        log_path.mkdir(parents=True, exist_ok=True)
        path = log_path.as_posix()
        logger.info(f"Created log directory at: {path}")
        return path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @abstractmethod
    def close(self):
        raise NotImplementedError("Subclasses must implement the close method.")

    @abstractmethod
    def report_hparams(self, category: str = "", *args: dict[str, Any], **kwargs):
        """
        Logs hyperparameters under the key: `category/<param_name>`.
        If a parameter is a dictionary, it is flattened.

        Args:
            category (str): The category under which to log the hyperparameters.
            *args (dict): Positional dictionaries of hyperparameters to log.
            **kwargs: Keyword arguments of hyperparameters to log.
        """
        raise NotImplementedError("Subclasses must implement the close method.")

    @abstractmethod
    def upload_code(self, file_path: str):
        raise NotImplementedError("Subclasses must implement the close method.")

    @abstractmethod
    def set_tags(self, **tags):
        """
        Sets tags for the current experiment.
        """
        raise NotImplementedError("Subclasses must implement the close method.")

    @abstractmethod
    def report_globals(self, metrics: dict[str, int | float]):
        """
        Logs a dictionary of final, global run metrics.
        For example, used to log final evaluation metrics.

        Args:
            metrics (dict[str, int | float]): The metrics to log.
        """
        raise NotImplementedError("Subclasses must implement the close method.")

    def report_scalars(self, scalers: dict[str, int | float] | dict[str, int | float | None], step: int, skip_infinite: bool = True):
        """
        Logs multiple scalar values.

        Args:
            scalers (dict[str, int | float | None]): The scalar values to log.
            step (int): The step number.
            skip_infinite (bool): Whether to skip logging if the value is not finite or NaN.
        """
        raise NotImplementedError("Subclasses must implement the close method.")

    @abstractmethod
    def report_scalar(self, tag: str, value: int | float | None, step: int, skip_infinite: bool = True):
        """
        Logs a scalar value.

        Args:
            tag (str): The name of the scalar value.
            value (int | float | None): The scalar value to log.
            step (int): The step number.
            skip_infinite (bool): Whether to skip logging if the value is not finite or NaN.
        """
        raise NotImplementedError("Subclasses must implement the close method.")

    @abstractmethod
    def report_image(self, tag: str, image: torch.Tensor | np.ndarray | PIL.Image.Image, step: int):
        """
        Logs an image.

        Args:
            tag (str): The name of the image.
            image (torch.Tensor | np.ndarray | PIL.Image.Image): The image to log, RGB format.
            step (int): The step number.
        """
        raise NotImplementedError("Subclasses must implement the close method.")
