from abc import abstractmethod, ABC
import torch
import random
import argparse
import sys
import time
from transformers import PreTrainedModel, PreTrainedTokenizer  # pyright: ignore[reportPrivateImportUsage]

from src.utils import env
from src.utils.logging import create_logger, setup_logging, loglevel_names
from src.data import TableLoader
from src.univ_attacks import UnivAttack
from src.adv_model import AdvModel
from src.config import GenConfig, StopCriteria
from src.eval import Evaluator
from src.metric_logger import MetricLogger

from scripts.utils.load_model import SUPPORTED_MODELS, load_model
from scripts.utils.load_dataset import SUPPORTED_DATASETS, load_dataset
from scripts.utils.load_evaluator import SUPPORTED_EVALUATORS, load_evaluators


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
            default="meta-llama/Llama-2-7b-chat-hf",
            metavar="MODEL",
            help=f"The model name to attack. Available models: {SUPPORTED_MODELS}",
        )

        parser.add_argument(
            "--dataset",
            type=str,
            choices=SUPPORTED_DATASETS,
            default="harmbench-std",
            metavar="DATASET",
            help=f"The datasets to use. Available datasets: {SUPPORTED_DATASETS}",
        )

        parser.add_argument(
            "--evaluator",
            type=str,
            nargs="+",
            choices=SUPPORTED_EVALUATORS,
            default=["strong-reject", "keyword-matching"],
            metavar="EVALUATOR",
            help=f"The attack evaluators to use. Available evaluators: {SUPPORTED_EVALUATORS}",
        )

        parser.add_argument(
            "--run_name",
            type=str,
            default=time.strftime("%Y-%m-%d_%H-%M-%S"),
            metavar="NAME",
            help="The name of the run, used for logging.",
        )

        parser.add_argument(
            "--train_batch",
            type=int,
            default=10,
            metavar="SIZE",
            help="The training batch size.",
        )

        parser.add_argument(
            "--eval_batch",
            type=int,
            default=25,
            metavar="SIZE",
            help="The evaluation batch size.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=random.randint(0, 1000000),
            help="Random seed for reproducibility.",
        )

        parser.add_argument(
            "--log_level",
            type=str,
            choices=loglevel_names(),
            default="INFO",
            metavar="LEVEL",
            help=f"Logging level to python-logger. Available levels: {loglevel_names()}",
        )

        attack_args = parser.add_argument_group("Base attack parameters")

        attack_args.add_argument(
            "--eval_metric",
            type=str,
            default=None,
            metavar="NAME",
            help="The evaluation metric to use for selecting the best adversarial prompt. "
            "If not specified, the default metric of the first evaluator will be used.",
        )

        attack_args.add_argument(
            "--eval_freq",
            type=float,
            default=1,
            metavar="NUM",
            help="Frequency of evaluation during training, in epochs. Can be a float.",
        )

        attack_args.add_argument(
            "--use_amp",
            choices=["true", "false"],
            metavar="BOOL",
            default="false",
            help="Whether to use automatic mixed precision (AMP) for training.",
        )

        gen_args = parser.add_argument_group("Generation parameters")

        gen_args.add_argument(
            "--do_sample",
            choices=["true", "false"],
            metavar="BOOL",
            default="true",
            help="Whether to use sampling for generation.",
        )

        gen_args.add_argument(
            "--temperature",
            type=float,
            default=None,
            metavar="TEMP",
            help="Sampling temperature for generation.",
        )

        gen_args.add_argument(
            "--top_p",
            type=float,
            default=None,
            metavar="P",
            help="Nucleus sampling top-p value for generation.",
        )

        gen_args.add_argument(
            "--top_k",
            type=int,
            default=None,
            metavar="K",
            help="Top-k sampling value for generation.",
        )

        gen_args.add_argument(
            "--max_new_tokens",
            type=int,
            default=512,
            metavar="NUM",
            help="Maximum number of new tokens to generate.",
        )

        stop_args = parser.add_argument_group("Stopping Criteria")

        stop_args.add_argument(
            "--max_time",
            type=int,
            default=120,
            metavar="MINUTES",
            help="The maximum training time in minutes.",
        )

        stop_args.add_argument(
            "--max_epochs",
            type=int,
            default=2000,
            metavar="NUM",
            help="The maximum number of training epochs.",
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
    def create_adversarial_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> AdvModel:
        pass

    @abstractmethod
    def initialize_attack(
        self,
        adv_model: AdvModel,
        evaluators: list[Evaluator],
        eval_metric: str | None,
        eval_freq: float,
        mixed_precision: bool,
        gen_config: GenConfig,
        metric_logger: MetricLogger,
    ) -> UnivAttack:
        pass

    def run(self):
        args = self.args()

        if not torch.cuda.is_available():
            logger.error("No GPU available. Exiting.")
            sys.exit(1)

        logger.info(f"Loading dataset: {args.dataset}")
        ds_train, ds_val, ds_test = load_dataset(args.dataset)
        dl_train = TableLoader(ds_train, batch_size=args.train_batch, shuffle=True)
        dl_eval = TableLoader(ds_val, batch_size=args.eval_batch, shuffle=False)
        dl_test = TableLoader(ds_test, batch_size=args.eval_batch, shuffle=False)

        logger.info(
            f"Loaded datasets with sample counts: "
            f"(train, val, test) = ({len(ds_train)}, {len(ds_val)}, {len(ds_test)})."
        )

        logger.info(f"Loading evaluator: {args.evaluator}")
        device_count = torch.cuda.device_count()
        gpus = [] if device_count <= 1 else list(range(device_count))[1:]
        logger.info(f"GPUs available for evaluators: {gpus}")
        evaluators = load_evaluators(args.evaluator, gpus=gpus)

        logger.info(f"Loading model: {args.model}")
        model, tokenizer = load_model(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
        adv_model = self.create_adversarial_model(model, tokenizer)
        logger.info(f"Model architecture: {adv_model.model}")

        root_dir = f"logs/{args.model.split('/')[-1]}/{args.dataset}"
        with MetricLogger(self.args().run_name, root_dir=root_dir, project="LLM-IML") as metric_logger:
            logger.info("Initializing attack...")

            gen_config = GenConfig(
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample.lower() == "true",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            univ_attack = self.initialize_attack(
                adv_model=adv_model,
                evaluators=evaluators,
                eval_metric=args.eval_metric,
                eval_freq=args.eval_freq,
                mixed_precision=args.use_amp.lower() == "true",
                gen_config=gen_config,
                metric_logger=metric_logger,
            )

            metric_logger.add_tags(
                model=args.model,
                num_tokens=adv_model.num_tokens,
                attack=type(univ_attack).__name__,
                dataset=", ".join(args.dataset),
                evaluators=", ".join(args.evaluator),
            )

            if main_file := getattr(sys.modules.get("__main__"), "__file__", None):
                metric_logger.log_code(main_file)
            if expr_file := getattr(sys.modules.get(__name__), "__file__", None):
                metric_logger.log_code(expr_file)

            if cm_task := metric_logger.cm_task:
                cm_task.register_artifact("train_data", dl_train.df, metadata=dl_train.get_hparams())
                cm_task.register_artifact("eval_data", dl_eval.df, metadata=dl_eval.get_hparams())
                cm_task.register_artifact("test_data", dl_test.df, metadata=dl_test.get_hparams())

            logger.info("Running attack...")

            stop = StopCriteria(
                max_epochs=args.max_epochs,
                max_time=args.max_time * 60,
            )

            adv_model = univ_attack.fit(dl_train, dl_eval, stop_criteria=stop)

            logger.info("Running test evaluation...")
            metrics = univ_attack.evaluate(evaluators, dl_test)
            metric_logger.log_metrics(metrics)
            if cm_task := metric_logger.cm_task:
                cm_task.upload_artifact(name="test_result", artifact_object=dl_eval.df)

        for eval in evaluators:
            eval.close()

    def main(self):
        try:
            self._parse_args()
            setup_logging(level=self.args().log_level)
            self.prepare_environment(seed=self.args().seed)
            self.run()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)
