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
        verbose: bool = True,
    ):
        """
        A base class for evaluators that processes batches of data and computes evaluation metrics.

        Args:
            name (str): Name of the evaluation method.
            required_columns (list[str]): List of required columns in the data for evaluation.
            verbose (bool): If True, enables verbose output during evaluation.
        """
        self.name = name
        self.verbose = verbose
        self.required_columns = required_columns

    @abstractmethod
    def get_hparams(self) -> dict:
        """
        Returns the hyperparameters of the evaluator as a dictionary.
        This method should be overridden by subclasses to provide specific hyperparameters.

        Returns:
            dict: Hyperparameters of the evaluator.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def process_batch(self, data: dict[str, list[Any]]) -> torch.Tensor:
        """
        Processes a batch of input and target texts, returning the evaluation metric.

        Args:
            eval_data (dict[str, list[Any]]): Tuple containing all relevant data to perform model evaluation

        Returns:
            torch.Tensor: Evaluation metric for each sample in the batch.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def evaluate(self, dl_eval: DF_Batcher) -> float:
        """
        Evaluates the model on the provided data loader using the generated outputs.  
        This methods sets a column `eval-{self.name}` in the data loader with the evaluation metric.

        Args:
            dl_eval (DF_Batcher): Data loader for evaluation.

        Returns:
            float: Average evaluation metric across all samples.
        """

        dl_eval.validate(self.required_columns)

        metrics = torch.zeros(dl_eval.n_samples, dtype=torch.float32)
        index = 0

        for batch_data in tqdm(dl_eval, desc=f"Evaluating {self.name}", disable=not self.verbose, leave=False):
            batch_metric = self.process_batch(batch_data).cpu()
            metrics[index:index + batch_metric.size(0)] = batch_metric
            index += batch_metric.size(0)
            
        dl_eval.set_column(self.name, metrics.tolist())
        return metrics.mean().item()


class MultiEvaluator(Evaluator):
    def __init__(
        self,
        evaluators: list[Evaluator],
        combine_fn: Callable[..., torch.Tensor],
        verbose: bool = False,
    ):
        """
        Combines multiple evaluators into a single evaluator, which combines
        their results using a specified combine function.

        Args:
            evaluators (list[Evaluator]): List of evaluators to combine.
            combine_fn (Callable[[..., torch.Tensor], torch.Tensor]): Function to combine the results of the evaluators.
                All arguments to this function should be tensors, and it should return a single tensor.
            verbose (bool): Whether to suppress verbose outputs and tqdm progress during evaluation.
        """
        name = " + ".join([evaluator.name for evaluator in evaluators])

        for evaler in evaluators:
            evaler.verbose = verbose

        super().__init__(
            name=name,
            required_columns=[],
            verbose=verbose,
        )

        self.evaluators = evaluators
        self.combine_fn = combine_fn

    def get_hparams(self) -> dict:
        """
        Returns the hyperparameters of the combined evaluator as a dictionary.
        This method combines the hyperparameters of all individual evaluators.

        Returns:
            dict: Combined hyperparameters of the evaluators.
        """
        hparams = {}
        for evaluator in self.evaluators:
            params = evaluator.get_hparams()
            params = {f"MultiEvaluator/{evaluator.name}/{k}": v for k, v in params.items()}
            hparams.update(params)
        return hparams

    def process_batch(self, data: dict[str, list[Any]]) -> torch.Tensor:
        """
        Processes a batch of input and target texts using multiple evaluators,
        and combines their results using the specified combine function.

        Args:
            data (dict[str, list[Any]]): Tuple containing all relevant data to perform model evaluation.

        Returns:
            torch.Tensor: Combined evaluation metric for each sample in the batch.
        """
        metrics = []
        for evaluator in self.evaluators:
            batch_metric = evaluator.process_batch(data)
            metrics.append(batch_metric.cpu())
        return self.combine_fn(*metrics)
