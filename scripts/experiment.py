from abc import abstractmethod, ABC
import torch
import random
import argparse
import logging
import sys

from src.utils import env
from src.utils.logging import create_logger, setup_logging
from src.data import DF_Batcher
from src.models import SUPPORTED_MODELS
from src.univ_attacks import UnivAttack


from src.adv_model import AdvModel
from src.config import StopCriteria
from src.eval import Evaluator

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

    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "--model",
            type=str,
            default="meta-llama/Llama-2-7b-chat-hf",
            help="The model name or path to use.",
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

        # verify valid model name
        if args.model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{args.model}'. Supported models: {SUPPORTED_MODELS}")

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

        torch.set_float32_matmul_precision("high")
        env.prepare_environment()
        env.set_seed(seed)
        logger.info(f"Random seed: {seed}")

    @abstractmethod
    def load_data(self, train_ratio: float) -> tuple[DF_Batcher, DF_Batcher]:
        pass

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

        logger.info("Loading data...")
        dl_train, dl_eval = self.load_data(train_ratio=args.train_ratio)
        logger.info(f"Train size: {dl_train.n_samples}")
        logger.info(f"Eval size: {dl_eval.n_samples}")

        logger.info("Loading evaluators...")
        evaluators = self.init_evaluators()
        logger.info(f"Evaluators loaded: {[ev.name for ev in evaluators]}")

        logger.info(f"Loading model: {args.model}")
        adv_model = self.init_model(model_name=args.model)

        logger.info("Initializing attack...")
        univ_attack = self.init_attack(adv_model, evaluators)

        stop = StopCriteria(max_epochs=1, max_time=60 * 60 * 10)

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
            self.prepare_environment(seed=self.args().seed)
            setup_logging(level=logging.WARNING if self.args().silent else logging.INFO)
            self.run()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)
