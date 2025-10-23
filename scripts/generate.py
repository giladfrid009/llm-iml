import torch
import random
import argparse
import sys
import pathlib


# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.eval import KeywordMatching
from src.config import GenConfig
from src.utils import env
from src.utils.logging import create_logger, setup_logging, loglevel_names
from src.data import TableLoader
from src.univ_attacks import UnivAttack
from src.adv_model import AdvModel
from src.metric_logger import MetricLogger

from scripts.utils.load_model import SUPPORTED_MODELS, load_model
from scripts.utils.load_dataset import SUPPORTED_DATASETS, load_dataset


logger = create_logger(__name__)


class Generator:
    def __init__(self) -> None:
        self._parsed_args = None

    def args(self) -> argparse.Namespace:
        if self._parsed_args is None:
            raise ValueError("Arguments have not been parsed yet. Call _parse_args() first.")
        return self._parsed_args

    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "embeds_path",
            type=str,
            metavar="PATH",
            help="Path to adversarial embeddings file (.pt)",
        )

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
            default="advbench",
            metavar="DATASET",
            help=f"The datasets to use. Available datasets: {SUPPORTED_DATASETS}",
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            default=50,
            metavar="SIZE",
            help="The evaluation batch size.",
        )

        parser.add_argument(
            "--include_train",
            action="store_true",
            help="Whether to generate on the training set as well.",
        )

        parser.add_argument(
            "--include_eval",
            action="store_true",
            help="Whether to generate on the evaluation set as well.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=random.randint(0, 1000000),
            help="Random seed for reproducibility.",
        )

        parser.add_argument(
            "--name_format",
            type=str,
            metavar="FMT",
            default="{model}_{dataset}_{split}.csv",
            help="Format string for naming the results files. Must include [{model}, {dataset}, {split}] placeholders.",
        )

        parser.add_argument(
            "--log_level",
            type=str,
            choices=loglevel_names(),
            default="INFO",
            metavar="LEVEL",
            help=f"Logging level to python-logger. Available levels: {loglevel_names()}",
        )

        adv_model_args = parser.add_argument_group("Adversarial model parameters")

        adv_model_args.add_argument(
            "--add_spaces",
            action="store_true",
            help="Whether to add spaces between the adversarial tokens.",
        )

        adv_model_args.add_argument(
            "--adv_prefix",
            action="store_true",
            help="Whether to add the adversarial tokens as a prefix (instead of suffix).",
        )

        gen_args = parser.add_argument_group("Generation parameters ")

        gen_args.add_argument(
            "--do_sample",
            action="store_true",
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

    def results_path(self, split: str) -> str:
        args = self.args()
        folder = pathlib.Path(args.embeds_path).parent / "generations"

        model_name = args.model.split("/")[-1].lower()
        dataset_name = args.dataset.lower()
        file_name = args.name_format.format(model=model_name, dataset=dataset_name, split=split)
        full_path: pathlib.Path = folder / file_name

        if not full_path.parent.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created results directory at: {full_path.parent.as_posix()}")

        if full_path.exists():
            raise FileExistsError(f"Results file already exists: {full_path.as_posix()}")

        return full_path.as_posix()

    def run(self):
        args = self.args()

        if not torch.cuda.is_available():
            logger.error("No GPU available. Exiting.")
            sys.exit(1)

        logger.info(f"Loading dataset: {args.dataset}")
        ds_train, ds_val, ds_test = load_dataset(args.dataset)
        dl_train = TableLoader(ds_train, batch_size=args.batch_size, shuffle=False)
        dl_eval = TableLoader(ds_val, batch_size=args.batch_size, shuffle=False)
        dl_test = TableLoader(ds_test, batch_size=args.batch_size, shuffle=False)

        logger.info(
            f"Loaded datasets with sample counts: "
            f"(train, val, test) = ({len(ds_train)}, {len(ds_val)}, {len(ds_test)})."
        )

        logger.info(f"Loading model: {args.model}")
        model, tokenizer = load_model(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")

        adv_embeds = torch.load(self.args().embeds_path, map_location=model.device)

        adv_model = AdvModel(
            model=model,
            tokenizer=tokenizer,
            num_tokens=adv_embeds.size(1),
            add_spaces=self.args().add_spaces,
            adv_suffix=not self.args().adv_prefix,
        )

        adv_model.set_embeddings(adv_embeds, strict=True)

        gen_config = GenConfig(
            max_new_tokens=self.args().max_new_tokens,
            do_sample=self.args().do_sample,
            temperature=self.args().temperature,
            top_p=self.args().top_p,
            remove_invalid_values=True,
        )

        univ_attack = UnivAttack(
            adv_model=adv_model,
            evaluators=[KeywordMatching()],
            eval_metric="Matching/GCG1",
            mixed_precision=False,
            metric_logger=MetricLogger(project="none", disabled=True),
            gen_config=gen_config,
        )

        if args.include_train:
            logger.info("Generating on training set...")
            univ_attack.predict(dl_train)
            result_path = self.results_path("train")
            dl_train.df.to_csv(result_path, index=False)
            logger.info(f"Saved training generations to {result_path}")

        if args.include_eval:
            logger.info("Generating on evaluation set...")
            univ_attack.predict(dl_eval)
            result_path = self.results_path("eval")
            dl_eval.df.to_csv(result_path, index=False)
            logger.info(f"Saved evaluation generations to {result_path}")

        logger.info("Generating on test set...")
        univ_attack.predict(dl_test)
        result_path = self.results_path("test")
        dl_test.df.to_csv(result_path, index=False)
        logger.info(f"Saved test generations to {result_path}")

    def main(self):
        try:
            self._parse_args()
            setup_logging(level=self.args().log_level)
            self.prepare_environment(seed=self.args().seed)
            self.run()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)


if __name__ == "__main__":
    Generator().main()
