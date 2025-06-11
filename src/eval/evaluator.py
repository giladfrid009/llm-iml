from abc import ABC, abstractmethod
from typing import Callable, List

import torch
from typing import Any
from src.data import DF_Batcher
from tqdm.auto import tqdm


class Evaluator(ABC):
    def __init__(
        self,
        name: str,
        required_columns: list[str],
        silent: bool = False,
    ):
        """
        Initializes the Evaluator with a name and a silent mode.

        Args:
            name (str): Name of the evaluation method.
            required_columns (list[str]): List of required columns in the data for evaluation.
            silent (bool): If True, suppresses output and tqdm during evaluation.
        """
        self.name = name
        self.silent = silent
        self.required_columns = required_columns

    @abstractmethod
    def process_batch(self, data: tuple[list[Any], ...]) -> torch.Tensor:
        """
        Processes a batch of input and target texts, returning the evaluation metric.

        Args:
            eval_data (tuple[list[Any], ...]): Tuple containing all relevant data to perform model evaluation

        Returns:
            torch.Tensor: Evaluation metric for each sample in the batch.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def evaluate(self, dl_eval: DF_Batcher) -> float:
        """
        Evaluates the model on the provided data loader using the generated outputs.

        Args:
            dl_eval (DF_Batcher): Data loader for evaluation.

        Returns:
            float: Average evaluation metric across all samples.
        """

        dl_eval.validate(self.required_columns)

        total_metric = 0.0
        num_samples = 0

        for batch_data in tqdm(dl_eval, desc=f"Evaluating {self.name}", disable=self.silent, leave=False):
            batch_metric = self.process_batch(batch_data)
            total_metric += batch_metric.sum().item()
            num_samples += batch_metric.size(0)

        return total_metric / num_samples if num_samples > 0 else 0.0


class MultiEvaluator(Evaluator):
    def __init__(
        self,
        evaluators: list[Evaluator],
        combine_fn: Callable[[list[torch.Tensor]], torch.Tensor],
        silent: bool = False,
    ):

        name = " + ".join([evaluator.name for evaluator in evaluators])

        super().__init__(
            name=name,
            required_columns=[],
            silent=silent,
        )

        self.evaluators = evaluators
        self.combine_fn = combine_fn

    def process_batch(self, data: tuple[list[Any], ...]) -> torch.Tensor:
        """
        Processes a batch of input and target texts using multiple evaluators,
        and combines their results using the specified combine function.

        Args:
            data (tuple[list[Any], ...]): Tuple containing all relevant data to perform model evaluation.

        Returns:
            torch.Tensor: Combined evaluation metric for each sample in the batch.
        """
        metrics = []
        for evaluator in self.evaluators:
            batch_metric = evaluator.process_batch(data)
            metrics.append(batch_metric)
        return self.combine_fn(metrics)
