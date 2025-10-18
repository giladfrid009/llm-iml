from abc import ABC, abstractmethod
from typing import Callable, Any
import numpy as np
from src.data import TableLoader
from tqdm.auto import tqdm
import inspect


class Evaluator(ABC):
    def __init__(self, name: str, metric_names: list[str], verbose: bool = True):
        """
        A base class for evaluators that processes batches of data and computes evaluation metrics.

        Args:
            name (str): Name of the evaluation method.
            metric_names (list[str]): List of metric names produced by the evaluator.
            verbose (bool): If True, enables verbose output during evaluation.
        """
        self.name = name
        self.metric_names = metric_names
        self.verbose = verbose

    def default_metric(self) -> str:
        """
        Returns the default metric name for this evaluator.

        Returns:
            str: Default metric name.
        """
        return self.metric_names[0]

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
    def close(self):
        """
        Cleans up any resources used by the evaluator.
        This method should be overridden by subclasses if they allocate resources that need to be released.
        """
        pass

    @abstractmethod
    def eval_batch(self, prompts: list[str], responses: list[str]) -> dict[str, list[float]]:
        """
        Processes a batch of input and target texts, returning the evaluation metric.

        Args:
            prompts (list[str]): List of input prompts for the model.
            responses (list[str]): List of model responses to evaluate.

        Returns:
            (dict[str, list[float]]): Evaluation metrics for each sample in the batch.
            Maps from metric name to list of metric values.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def evaluate(self, dl_eval: TableLoader) -> dict[str, float]:
        """
        Evaluates the model on the provided data loader using the generated outputs.
        This methods sets new columns in the data loader for each metric in `self.metric_names`.

        Args:
            dl_eval (TableLoader): Data loader for evaluation.

        Returns:
            dict[str, float]: Average evaluation metrics over the entire dataset.
        """

        dl_eval.validate(["prompt", "response"])

        metrics: dict[str, list[float]] = {k: [] for k in self.metric_names}
        for batch_data in tqdm(dl_eval, desc=f"Evaluating {self.name}", disable=not self.verbose, leave=False):
            batch_prompts = [str(s) for s in batch_data["prompt"]]
            batch_responses = [str(s) for s in batch_data["response"]]
            batch_metric = self.eval_batch(batch_prompts, batch_responses)
            for metric_name, metric_values in batch_metric.items():
                metrics[metric_name].extend(metric_values)

        for metric_name, metric_values in metrics.items():
            dl_eval.set_column(metric_name, metric_values)

        # Compute averages, ignoring non-finite values
        numpy_metrics = dl_eval.df[self.metric_names].to_numpy(dtype=float)
        numpy_metrics = np.where(np.isfinite(numpy_metrics), numpy_metrics, np.nan)
        averages = np.nanmean(numpy_metrics, axis=0).flatten().tolist()

        return {k: v for k, v in zip(self.metric_names, averages)}


class MultiEvaluator(Evaluator):
    def __init__(
        self,
        evaluators: list[Evaluator],
        combine_fn: Callable[[dict[str, float]], float],
        verbose: bool = True,
    ):
        """
        Combines multiple evaluators into a single evaluator, which combines
        their results using a specified combine function.

        Args:
            evaluators (list[Evaluator]): List of evaluators to combine.
            combine_fn (Callable[dict[str, float], float]): Function to combine the metrics from the individual evaluators.
                Receives a dictionary mapping metric names to their values and returns a single float value.
            verbose (bool): Whether to suppress verbose outputs and tqdm progress during evaluation.
        """

        name = f"MultiEval({','.join([ev.name for ev in evaluators])})"

        for ev in evaluators:
            ev.verbose = verbose

        metric_names = [name]
        for ev in evaluators:
            metric_names.extend(ev.metric_names)

        super().__init__(name=name, metric_names=metric_names, verbose=verbose)

        self.evaluators = evaluators
        self.combine_fn = combine_fn

    def default_metric(self) -> str:
        return self.name
    
    def get_hparams(self) -> dict:
        hparams: dict[str, Any] = {
            "inner_evaluators": str([ev.name for ev in self.evaluators]),
            "combine_fn": inspect.getsource(self.combine_fn),
            "metrics": str(self.metric_names),
        }
        hparams.update({ev.name: ev.get_hparams() for ev in self.evaluators})
        return hparams

    def eval_batch(self, prompts: list[str], responses: list[str]) -> dict[str, list[float]]:
        metrics: dict[str, list[float]] = {}
        for ev in self.evaluators:
            batch_metric = ev.eval_batch(prompts, responses)
            metrics.update(batch_metric)

        results = []
        for i in range(len(prompts)):
            metric_dict = {k: v[i] for k, v in metrics.items()}
            value = self.combine_fn(metric_dict)
            results.append(value)

        metrics[self.name] = results
        return metrics

    def close(self):
        for ev in self.evaluators:
            ev.close()
