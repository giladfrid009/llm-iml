import torch
import random
import argparse
import os
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
from src.utils.trackers import MetricTracker

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
            "embeds",
            type=str,
            nargs="+",
            metavar="PATH",
            help="Paths to adversarial embeddings files (.pt)",
        )

        parser.add_argument(
            "--model",
            type=str,
            metavar="MODEL",
            help=f"The model name to attack. Supported models: {SUPPORTED_MODELS}",
        )

        parser.add_argument(
            "--dataset",
            type=str,
            nargs="+",
            choices=SUPPORTED_DATASETS,
            default=["advbench"],
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
            "--splits",
            type=str,
            nargs="+",
            choices=["train", "val", "test"],
            default=["train"],
            metavar="SPLIT",
            help="Dataset splits to generate. Only the named output files are touched.",
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
            "--overwrite",
            action="store_true",
            help="Whether to overwrite existing results files. If not set, the script will raise an error if a results file already exists for a given model/dataset/split.",
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

        self._parsed_args = parser.parse_args()
        args = self.args()

        if len(args.splits) != len(set(args.splits)):
            parser.error("--splits must not contain duplicate split names")

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

    def results_path(self, embeds_path: str, dataset: str, split: str) -> str:
        args = self.args()
        folder = pathlib.Path(embeds_path).parent / "generations"

        model_name = args.model.split("/")[-1].lower()
        dataset_name = dataset.lower()
        file_name = args.name_format.format(model=model_name, dataset=dataset_name, split=split)
        full_path: pathlib.Path = folder / file_name

        if not full_path.parent.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created results directory at: {full_path.parent.as_posix()}")

        if full_path.exists() and not args.overwrite:
            raise FileExistsError(f"Results file already exists: {full_path.as_posix()}")

        return full_path.as_posix()

    def run(self):
        args = self.args()

        if not torch.cuda.is_available():
            logger.error("No GPU available. Exiting.")
            sys.exit(1)

        logger.info(f"Loading model: {args.model}")
        model, tokenizer = load_model(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")

        gen_config = GenConfig(
            max_new_tokens=self.args().max_new_tokens,
            do_sample=self.args().do_sample == "true",
            temperature=self.args().temperature,
            top_p=self.args().top_p,
            remove_invalid_values=True,
        )

        datasets = {}
        for dataset in args.dataset:
            logger.info(f"Loading dataset: {dataset}")
            ds_train, ds_val, ds_test = load_dataset(dataset)
            logger.info(f"Loaded dataset {dataset} with sample counts: (train, val, test) = ({len(ds_train)}, {len(ds_val)}, {len(ds_test)}).")
            datasets[dataset] = {"train": ds_train, "val": ds_val, "test": ds_test}

        for embeds_path in args.embeds:
            logger.info(f"Loading adversarial embeddings: {embeds_path}")
            adv_embeds = torch.load(embeds_path, map_location=model.device)
            
            adv_model = AdvModel(
                model=model,
                tokenizer=tokenizer,
                num_tokens=adv_embeds.size(1),
                add_spaces=self.args().add_spaces,
                adv_suffix=not self.args().adv_prefix,
            )
            
            adv_model.set_embeddings(adv_embeds, strict=True)
            
            univ_attack = UnivAttack(
                adv_model=adv_model,
                evaluators=[KeywordMatching()],
                eval_metric="Matching/GCG1",
                metric_tracker=MetricTracker.create(kind="clearml", project="none", disabled=True),
                gen_config=gen_config,
            )

            for dataset, split_datasets in datasets.items():
                for split in args.splits:
                    try:
                        split_path = self.results_path(embeds_path, dataset, split)
                        logger.info(f"Generating on {split} set...")
                        loader = TableLoader(split_datasets[split], batch_size=args.batch_size, shuffle=False)
                        univ_attack.predict(loader)
                        loader.df.to_csv(split_path, index=False)
                        logger.info(f"Saved {split} generations to {split_path}")

                    except FileExistsError as e:
                        logger.warning(str(e))
                        continue

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
