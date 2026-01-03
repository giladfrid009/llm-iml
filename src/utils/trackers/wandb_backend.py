import pathlib
import wandb
from wandb import Run as WandbRun
from typing import Any
import numpy as np
import PIL.Image
import torch

from src.utils.logging import create_logger
from src.utils.trackers.base import MetricTracker


logger = create_logger(__name__)


class WandbTracker(MetricTracker):
    def __init__(
        self,
        *names: str,
        project: str,
        root_dir: str = "logs",
        disabled: bool = False,
        **kwargs,
    ):
        super().__init__(*names, project=project, root_dir=root_dir, disabled=disabled)

        self.wandb_run: WandbRun | None = None

        if not disabled:
            self.wandb_run = wandb.init(
                project=self.project,
                dir=self.log_dir,
                name=self.run_name,
                allow_val_change=True,
                sync_tensorboard=False,
                monitor_gym=False,
                **kwargs,
            )

    def close(self):
        if self.wandb_run is not None:
            self.wandb_run.finish(exit_code=0)
            self.wandb_run = None  # type: ignore

    def report_hparams(self, category: str = "", *args: dict[str, Any], **kwargs):
        if self.wandb_run is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        hparams = {}
        for d in args:
            hparams.update(d)
        hparams.update(kwargs)

        if category:
            hparams = {category: hparams}

        self.wandb_run.config.update(hparams, allow_val_change=True)

    def upload_code(self, file_path: str):
        if self.wandb_run is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        if not pathlib.Path(file_path).is_file():
            logger.warning(f"File '{file_path}' does not exist. Cannot log code.")
            return

        file_name = pathlib.Path(file_path).name
        upload_result = self.wandb_run.log_code(
            name=file_name,
            include_fn=lambda path: pathlib.Path(path) == pathlib.Path(file_path),
        )

        if not upload_result:
            logger.warning(f"Failed to log code file '{file_path}'.")

    def set_tags(self, **tags):
        if self.wandb_run is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        if len(tags) == 0:
            logger.warning("No tags provided.")
            return

        tags = {k.title(): str(v) for k, v in tags.items()}
        formatted = tuple(f"{tag}: {text}" for tag, text in tags.items())

        if self.wandb_run.tags is None:
            self.wandb_run.tags = formatted
        else:
            self.wandb_run.tags += formatted

    def report_globals(self, metrics: dict[str, int | float]):
        if self.wandb_run is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        table = wandb.Table(columns=["Metric", "Value"], log_mode="INCREMENTAL")
        for k, v in metrics.items():
            table.add_data(k, float(v))

        self.wandb_run.log({"Global-Metrics": table})
        metrics = {f"Global/{k}": v for k, v in metrics.items()}
        self.wandb_run.summary.update(metrics)

    def report_scalars(self, scalers: dict[str, int | float] | dict[str, int | float | None], step: int, skip_infinite: bool = True):
        if self.wandb_run is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        if skip_infinite:
            scalers = {k: v for k, v in scalers.items() if v is not None and np.isfinite(v)}

        scalers = {k: (v if v is not None else float("nan")) for k, v in scalers.items()}

        self.wandb_run.log(data=scalers, step=step)

    def report_scalar(self, tag: str, value: int | float | None, step: int, skip_infinite: bool = True):
        if self.wandb_run is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        if skip_infinite and (value is None or not np.isfinite(value)):
            return  # skip logging non-finite values

        if value is None:
            value = float("nan")

        self.wandb_run.log(data={tag: value}, step=step)

    def report_image(self, tag: str, image: torch.Tensor | np.ndarray | PIL.Image.Image, step: int):
        if self.wandb_run is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        title = tag.split("/", maxsplit=1)[0] if "/" in tag else tag
        title = title.title()

        if isinstance(image, torch.Tensor):
            image = image.numpy(force=True)

        wandb_image = wandb.Image(image, caption=title)
        self.wandb_run.log({tag: wandb_image}, step=step)
