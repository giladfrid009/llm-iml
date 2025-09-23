from abc import ABC, abstractmethod

from src.data import DF_Batcher
from tqdm.auto import tqdm


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
            dict[str, list[float]]: Evaluation metrics for each sample in the batch.
            Maps from metric name to list of metric values.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def evaluate(self, dl_eval: DF_Batcher) -> dict[str, float]:
        """
        Evaluates the model on the provided data loader using the generated outputs.
        This methods sets a column `eval-{self.name}` in the data loader with the evaluation metric.

        Args:
            dl_eval (DF_Batcher): Data loader for evaluation.

        Returns:
            dict[str, float]: Average evaluation metrics over the entire dataset.
        """

        dl_eval.validate(["prompt", "response"])

        metrics = {k: [] for k in self.metric_names}
        for batch_data in tqdm(dl_eval, desc=f"Evaluating {self.name}", disable=not self.verbose, leave=False):
            batch_metric = self.eval_batch(batch_data["prompt"], batch_data["response"])
            for metric_name, metric_values in batch_metric.items():
                metrics[metric_name].extend(metric_values)

        for metric_name, metric_values in metrics.items():
            dl_eval.set_column(metric_name, metric_values)

        averages = {metric_name: sum(values) / len(values) for metric_name, values in metrics.items()}
        return averages
