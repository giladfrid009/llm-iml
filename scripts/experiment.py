from abc import abstractmethod, ABC
import torch
import random
import argparse
import logging
import sys
import pandas as pd
import os
import time

from src.utils import env
from src.utils.logging import create_logger, setup_logging
from src.data import TableLoader
from src.models import SUPPORTED_MODELS
from src.univ_attacks import UnivAttack
from src.adv_model import AdvModel
from src.config import StopCriteria
from src.eval import Evaluator


SUPPORTED_DATASETS = [
    "adv_bench",
    "harm_bench",
    "jailbreak_bench",
    "malicious_instruct",
]


logger = create_logger(__name__)


class Experiment(ABC):
    def __init__(self) -> None:
        self._parsed_args = None

    def args(self) -> argparse.Namespace:
        if self._parsed_args is None:
            raise ValueError("Arguments have not been parsed yet. Call _parse_args() first.")
        return self._parsed_args

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Override to add custom command line arguments."""
        pass

    # TODO: add run_name optional argument
    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "--model",
            type=str,
            choices=SUPPORTED_MODELS,
            default="meta-llama/Llama-2-7b-chat-hf",
            help="The model name or path to use.",
        )

        parser.add_argument(
            "--run_name",
            type=str,
            default=time.strftime("%Y-%m-%d_%H-%M-%S"),
            help="The name of the run, used for logging.",
        )

        parser.add_argument(
            "--dataset",
            type=str,
            choices=SUPPORTED_DATASETS,
            default="harm_bench",
            help="The dataset to use.",
        )

        parser.add_argument(
            "--train_ratio",
            type=str,
            default=0.65,
            help="The ratio of training data to use.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=random.randint(0, 1000000),
            help="Random seed for reproducibility (default: random).",
        )

        parser.add_argument(
            "--silent",
            action="store_true",
            help="Disable verbose outputs.",
        )

        self.add_arguments(parser)
        self._parsed_args = parser.parse_args()
        args = self.args()

        # print the parsed arguments
        print()
        print("Parsed arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print()

        return args

    def prepare_environment(self, seed: int | None):
        if seed is None:
            seed = random.randint(0, 10000)
        logger.info(f"Random seed: {seed}")

        torch.set_float32_matmul_precision("high")
        env.prepare_environment()
        env.set_seed(seed)

    def load_data(self, dataset_name: str, train_ratio: float) -> tuple[TableLoader, TableLoader]:
        data_path = f"data/{dataset_name}/harmful_behaviors.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = pd.read_csv(data_path)
        data = data.rename(columns={"goal": "prompt"})
        data = data.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle

        split = int(train_ratio * len(data))
        dl_train = TableLoader(data.iloc[:split].copy(), batch_size=10, shuffle=True)
        dl_eval = TableLoader(data.iloc[split:].copy(), batch_size=25, shuffle=False)

        logger.info(f"Train size: {dl_train.n_samples}")
        logger.info(f"Eval size: {dl_eval.n_samples}")

        return dl_train, dl_eval

    @abstractmethod
    def init_evaluators(self) -> list[Evaluator]:
        pass

    @abstractmethod
    def init_model(self, model_name: str) -> AdvModel:
        pass

    @abstractmethod
    def init_attack(self, adv_model: AdvModel, evaluators: list[Evaluator]) -> UnivAttack:
        pass

    def run(self):
        args = self.args()

        logger.info(f"Loading dataset: {args.dataset}")
        dl_train, dl_eval = self.load_data(args.dataset, train_ratio=args.train_ratio)

        logger.info("Loading evaluators...")
        evaluators = self.init_evaluators()
        logger.info(f"Evaluators loaded: {[ev.name for ev in evaluators]}")

        logger.info(f"Loading model: {args.model}")
        adv_model = self.init_model(args.model)
        logger.info(f"Model architecture: {adv_model.model}")

        logger.info("Initializing attack...")
        univ_attack = self.init_attack(adv_model, evaluators)

        logger.info("Logging experiment data...")
        if main_file := getattr(sys.modules.get("__main__"), "__file__", None):
            univ_attack.metric_logger.log_code(main_file)
        if expr_file := getattr(sys.modules.get(__name__), "__file__", None):
            univ_attack.metric_logger.log_code(expr_file)

        univ_attack.metric_logger.cm_task.register_artifact("train_data", dl_train.df, metadata=dl_train.get_hparams())
        univ_attack.metric_logger.cm_task.register_artifact("eval_data", dl_eval.df, metadata=dl_eval.get_hparams())

        stop = StopCriteria(max_epochs=2000, max_time=60 * 60 * 3)

        try:
            logger.info("Running attack...")
            adv_model = univ_attack.fit(dl_train, dl_eval, stop_criteria=stop)

            logger.info("Running final eval...")
            metrics = univ_attack.evaluate(adv_model, evaluators, dl_eval)
            univ_attack.metric_logger.log_metrics(metrics)
            univ_attack.metric_logger.cm_task.upload_artifact(name="eval_result", artifact_object=dl_eval.df)

        finally:
            univ_attack.close()
            for eval in evaluators:
                eval.close()

    def main(self):
        try:
            self._parse_args()
            setup_logging(level=logging.WARNING if self.args().silent else logging.INFO)
            self.prepare_environment(seed=self.args().seed)
            self.run()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)
