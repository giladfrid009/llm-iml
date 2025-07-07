import torch
import pathlib
import time
from typing import TYPE_CHECKING
import logging
from torch.utils.tensorboard.writer import SummaryWriter

# Handle gracefully import failures

if TYPE_CHECKING:
    import clearml

try:
    from clearml import Task
except ImportError:
    Task = None


class Logger:
    def __init__(self, root_dir: str | None = None):
        """
        Base class for logging metrics and visualizations during training.
        """
        self.root_dir = root_dir
        self.tb_writer: SummaryWriter | None = None
        self.cm_task: clearml.Task | None = None
        self.global_step: int = 0
        self._hparams: dict[str, object] = {}

    def register_hparams(self, hparams: dict[str, object]):
        """
        Registers hyperparameters to be logged.

        Args:
            hparams (dict[str, object]): The hyperparameters to log.
        """
        clean_hparams = {}
        for k, v in hparams.items():
            if isinstance(v, dict):
                self.register_hparams({f"{k}/{k2}": v2 for k2, v2 in v.items()})
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = v.item()
            elif isinstance(v, tuple):
                v = str(v)

            if v is None or isinstance(v, (int, float, str, bool)):
                clean_hparams[k] = v

        self._hparams.update(clean_hparams)

    def initialize(self, *subdir_parts: str):
        """
        Initializes the logger by creating a new log directory and setting up TensorBoard and ClearML logging.

        Args:
            *subdir_parts (str): Log sub-directory parts, under the root directory.
        """
        if self.root_dir is None:
            return None

        self.close()

        # Create log directory
        sub_dir = pathlib.Path(*subdir_parts)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = pathlib.Path(self.root_dir) / sub_dir / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging enabled. Saving logs to: {log_dir}")

        # Init TensorBoard
        self.global_step = 0
        self.tb_writer = SummaryWriter(log_dir=log_dir)

        if Task is None:
            self.cm_task = None
            logging.warning("ClearML installation not found. ClearML logging disabled.")
            return

        # Init ClearML
        task_name = str.join(" - ", sub_dir.parts) + f" - {timestamp}"
        self.cm_task = Task.init(project_name="LLM-IML", task_name=task_name)

    def log_dir(self) -> str | None:
        if self.tb_writer is not None:
            return str(self.tb_writer.log_dir)
        return None

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None

        if self.cm_task is not None:
            self.cm_task.flush()
            self.cm_task.close()
            self.cm_task = None

    def __del__(self):
        self.close()

    def step(self, n: int = 1):
        """
        Increments the global step counter.
        """
        self.global_step += n

    def log_hparams(self):
        if self.tb_writer is not None:
            self.tb_writer.add_hparams(self._hparams, {}, run_name=".")

        if self.cm_task is not None:
            self.cm_task.update_parameters(self._hparams)

    def log_metrics(self, metrics: dict[str, int | float]):
        """
        Logs a dictionary of global run metrics.
        For example, used to log final evaluation metrics.

        Args:
            metrics (dict[str, int | float]): The metrics to log.
        """
        if self.tb_writer is not None:
            formatted = {f"final/{k}": v for k, v in metrics.items()}
            self.tb_writer.add_hparams({}, metric_dict=formatted, run_name=".")

    def log_scalers(self, scalers: dict[str, int | float | torch.Tensor | None], step: int | None = None):
        for key, value in scalers.items():
            self.log_scalar(key, value, step)

    def log_scalar(self, tag: str, value: int | float | torch.Tensor | None, step: int | None = None):
        """
        Logs a scalar value.

        Args:
            tag (str): The name of the scalar value.
            value (int | float | torch.Tensor | None): The scalar value to log.
            step (int, optional): The step number. If None, uses the global step counter.
        """
        if value is None or self.tb_writer is None:
            return

        if step is None:
            step = self.global_step

        if isinstance(value, torch.Tensor):
            value = value.item()

        self.tb_writer.add_scalar(tag, value, step)

    def log_image(self, tag: str, img: torch.Tensor, step: int | None = None):
        """
        Logs an image.

        Args:
            tag (str): The name of the image.
            img (torch.Tensor): The image tensor to log.
            step (int, optional): The step number. If None, uses the global step counter.
        """
        if self.tb_writer is None:
            return

        if step is None:
            step = self.global_step

        img = img.cpu().detach().clamp(0, 1)
        if img.ndim == 4:
            img = img[0]

        self.tb_writer.add_image(tag, img, step)

    def add_graph(self, model: torch.nn.Module, input: torch.Tensor):
        """
        Logs the model graph.

        Args:
            model (torch.nn.Module): The model to log.
            input_shape (tuple[int]): The shape of the input tensor.
        """
        if self.tb_writer is None:
            return

        self.tb_writer.add_graph(model, input)

    def add_tags(self, **tags):
        """
        Adds tags to Tensorboard and the ClearML task.

        Args:for
            *tags: The tags to add.
        """
        if len(tags) == 0:
            return

        tags = {k.title(): str(v) for k, v in tags.items()}

        if self.tb_writer is not None:
            for tag, text in tags.items():
                self.tb_writer.add_text(tag, text)

        if self.cm_task is not None:
            formatted = [f"{tag}: {text}" for tag, text in tags.items()]
            self.cm_task.add_tags(formatted)
