import pathlib
from clearml import Task
from typing import Any
import numpy as np
import PIL.Image
import torch

from src.utils.logging import create_logger
from src.utils.trackers.base import MetricTracker


logger = create_logger(__name__)


class ClearmlTracker(MetricTracker):
    def __init__(
        self,
        *names: str,
        project: str,
        root_dir: str = "logs",
        disabled: bool = False,
        **kwargs,
    ):
        super().__init__(*names, project=project, root_dir=root_dir, disabled=disabled)

        self.cm_task: Task | None = None

        if not disabled:
            Task.set_random_seed(None)  # NOTE: are you kidding me
            self.cm_task = Task.init(
                project_name=project,
                task_name=self.run_name,
                reuse_last_task_id=False,
                auto_resource_monitoring={
                    "first_report_sec": 1800.0,
                    "sample_frequency_per_sec": 1,
                },
                **kwargs,
            )

    def close(self):
        if self.cm_task is not None:
            self.cm_task.flush(wait_for_uploads=True)
            self.cm_task.close()
            self.cm_task = None  # type: ignore

    def report_hparams(self, category: str = "", *args: dict[str, Any], **kwargs):
        if self.cm_task is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        hparams = flatten_dict(kwargs, category)
        for d in args:
            hparams.update(flatten_dict(d, category))

        self.cm_task.update_parameters(hparams)

    def upload_code(self, file_path: str):
        if self.cm_task is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        if not pathlib.Path(file_path).is_file():
            logger.warning(f"File '{file_path}' does not exist. Cannot log code.")
            return

        file_name = pathlib.Path(file_path).name
        file_content = pathlib.Path(file_path).read_text()
        upload_result = self.cm_task.upload_artifact(
            name=file_name,
            artifact_object=file_path,
            preview=file_content,
            metadata={"type": "code", "full_path": file_path},
        )

        if not upload_result:
            logger.warning(f"Failed to log code file '{file_path}'.")

    def set_tags(self, **tags):
        if self.cm_task is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        if len(tags) == 0:
            logger.warning("No tags provided.")
            return

        tags = {k.title(): str(v) for k, v in tags.items()}
        formatted = [f"{tag}: {text}" for tag, text in tags.items()]
        self.cm_task.add_tags(formatted)

    def report_globals(self, metrics: dict[str, int | float]):
        if self.cm_task is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        for k, v in metrics.items():
            self.cm_task.logger.report_single_value(k, v)

    def report_scalars(self, scalers: dict[str, int | float] | dict[str, int | float | None], step: int, skip_infinite: bool = True):
        if self.cm_task is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        for key, value in scalers.items():
            self.report_scalar(key, value, step, skip_infinite)

    def report_scalar(self, tag: str, value: int | float | None, step: int, skip_infinite: bool = True):
        if self.cm_task is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        if "/" in tag:
            split = tag.split("/", maxsplit=1)
            title, series = split[0], split[1]
        else:
            title = series = tag

        title = title.title()

        if skip_infinite and (value is None or not np.isfinite(value)):
            return  # skip logging non-finite values

        if value is None:
            value = float("nan")

        cm_logger = self.cm_task.get_logger()
        cm_logger.report_scalar(title=title, series=series, value=float(value), iteration=step)

    def report_image(self, tag: str, image: torch.Tensor | np.ndarray | PIL.Image.Image, step: int):
        if self.cm_task is None or self.disabled:
            logger.debug("Tracker is disabled. Skipping.")
            return

        if "/" in tag:
            split = tag.split("/", maxsplit=1)
            title, series = split[0], split[1]
        else:
            title = series = tag

        title = title.title()
        series = series.title()

        if isinstance(image, torch.Tensor):
            image = image.numpy(force=True)

        cm_logger = self.cm_task.get_logger()
        cm_logger.report_image(title=title, series=series, image=image, iteration=step, max_image_history=-1)
