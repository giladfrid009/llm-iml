from abc import abstractmethod, ABC
import torch
import random
import argparse
import logging
import sys
import time
from transformers import PreTrainedModel, PreTrainedTokenizer  # pyright: ignore[reportPrivateImportUsage]

from src.utils import env
from src.utils.logging import create_logger, setup_logging
from src.data import TableLoader
from src.univ_attacks import UnivAttack
from src.adv_model import AdvModel
from src.config import StopCriteria
from src.eval import Evaluator

from scripts.load_model import SUPPORTED_MODELS, load_model
from scripts.load_dataset import SUPPORTED_DATASETS, load_datasets


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
            choices=SUPPORTED_MODELS,
            required=True,
            help="The model name to attack.",
        )

        parser.add_argument(
            "--dataset",
            type=str,
            nargs="+",
            choices=SUPPORTED_DATASETS,
            required=True,
            help="The dataset(s) to use. If multiple datasets are provided, they will be concatenated.",
        )

        parser.add_argument(
            "--run_name",
            type=str,
            default=time.strftime("%Y-%m-%d_%H-%M-%S"),
            help="The name of the run, used for logging.",
        )

        parser.add_argument(
            "--val_ratio",
            type=str,
            default=0.35,
            help="The ratio of the validation set.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=random.randint(0, 1000000),
            help="Random seed for reproducibility.",
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

    @abstractmethod
    def create_evaluators(self) -> list[Evaluator]:
        pass

    @abstractmethod
    def create_adversarial_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> AdvModel:
        pass

    @abstractmethod
    def initialize_attack(self, adv_model: AdvModel, evaluators: list[Evaluator]) -> UnivAttack:
        pass

    def run(self):
        args = self.args()

        logger.info(f"Loading dataset(s): {args.dataset}")
        ds_train, ds_val, ds_test = load_datasets(*args.dataset, val_ratio=args.val_ratio)
        dl_train = TableLoader(ds_train, batch_size=10, shuffle=True)
        dl_eval = TableLoader(ds_val, batch_size=25, shuffle=False)
        dl_test = TableLoader(ds_test, batch_size=25, shuffle=False)
        logger.info(
            f"Loaded datasets with sample counts: "
            f"train={len(dl_train.df)}, val={len(dl_eval.df)}, test={len(dl_test.df)}"
        )

        logger.info("Loading evaluators...")
        evaluators = self.create_evaluators()
        logger.info(f"Evaluators loaded: {[ev.name for ev in evaluators]}")

        logger.info(f"Loading model: {args.model}")
        model, tokenizer = load_model(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
        adv_model = self.create_adversarial_model(model, tokenizer)
        logger.info(f"Model architecture: {adv_model.model}")

        logger.info("Initializing attack...")
        univ_attack = self.initialize_attack(adv_model, evaluators)

        logger.info("Logging experiment data...")
        if main_file := getattr(sys.modules.get("__main__"), "__file__", None):
            univ_attack.metric_logger.log_code(main_file)
        if expr_file := getattr(sys.modules.get(__name__), "__file__", None):
            univ_attack.metric_logger.log_code(expr_file)

        univ_attack.metric_logger.cm_task.register_artifact("train_data", dl_train.df, metadata=dl_train.get_hparams())
        univ_attack.metric_logger.cm_task.register_artifact("eval_data", dl_eval.df, metadata=dl_eval.get_hparams())
        univ_attack.metric_logger.cm_task.register_artifact("test_data", dl_test.df, metadata=dl_test.get_hparams())

        univ_attack.metric_logger.add_tags(
            model=args.model,
            num_tokens=adv_model.num_tokens,
            attack=type(univ_attack).__name__,
            dataset=",".join(args.dataset),
        )

        stop = StopCriteria(max_epochs=2000, max_time=60 * 60 * 3)

        try:
            logger.info("Running attack...")
            adv_model = univ_attack.fit(dl_train, dl_eval, stop_criteria=stop)

            logger.info("Running test evaluation...")
            metrics = univ_attack.evaluate(adv_model, evaluators, dl_test)
            univ_attack.metric_logger.log_metrics(metrics)
            univ_attack.metric_logger.cm_task.upload_artifact(name="test_result", artifact_object=dl_eval.df)

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
