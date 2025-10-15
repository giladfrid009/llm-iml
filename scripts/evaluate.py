import pandas as pd
import sys
import pathlib
import argparse
import random
import torch
import logging

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from gserve.configs import ServeConfig, LLMConfig
from scripts.utils.load_evaluator import load_single_evaluator, SUPPORTED_EVALUATORS
from src.utils.logging import create_logger, setup_logging, loglevel_names, parse_log_level
from src.utils.torch import clear_memory
from src.eval import Evaluator
from src.data import TableLoader
from src.utils import env


logger = create_logger(__name__)


def _create_serve(gpu_id: int, loglevel: str) -> ServeConfig:
    verbose = parse_log_level(loglevel) <= logging.DEBUG
    return ServeConfig(gpu_ids=[gpu_id], startup_timeout=20 * 60, client_timeout=60, verbose=verbose)


def _display_width() -> int:
    try:
        import shutil

        return shutil.get_terminal_size().columns
    except Exception:
        return 120


def _read_df(path: pathlib.Path, **kwargs) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, lines=False, **kwargs)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".feather":
        return pd.read_feather(path, **kwargs)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path, **kwargs)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        nargs="+",
        required=True,
        help="Paths to evaluation data CSV file.",
    )

    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=SUPPORTED_EVALUATORS,
        help="List of evaluator names to run. If not provided, all evaluators will be run.",
        choices=SUPPORTED_EVALUATORS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for data loading.",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use for evaluation.",
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

    args = parser.parse_args()

    # print the parsed arguments
    print()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    return args


def read_data(paths: list[str]) -> tuple[list[str], list[pd.DataFrame]]:
    path_list = []
    data_list = []

    for path in paths:
        path = pathlib.Path(path)

        if path.is_dir():
            logger.info(f"Loading all files from directory: {path}")
            sub_paths, sub_data = read_data([p.as_posix() for p in path.iterdir() if p.is_file()])
            logger.info(f"Loaded {len(sub_data)} files from directory: {path}")
            path_list.extend(sub_paths)
            data_list.extend(sub_data)
            continue

        try:
            df = _read_df(path)
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}. Skipping...")
            continue

        req_cols = {"prompt", "response"}
        if not req_cols.issubset(df.columns):
            logger.error(f"File {path} is missing required columns: {req_cols}. Skipping...")

        path_list.append(path.as_posix())
        data_list.append(df)

    return path_list, data_list


def prepare_environment(seed: int | None):
    if seed is None:
        seed = random.randint(0, 10000)
    logger.info(f"Random seed: {seed}")

    torch.set_float32_matmul_precision("high")
    env.prepare_environment()
    env.set_seed(seed)


def main(args: argparse.Namespace):
    # get terminal width for pretty printing
    width = _display_width()

    # read data
    path_list, data_list = read_data(args.path)
    loader_list = [TableLoader(df, batch_size=args.batch_size, shuffle=False) for df in data_list]
    all_results = {ds_name: {} for ds_name in path_list}

    for eval_name in args.names:
        print()
        print("=".center(width, "="))
        print(f"Running evaluator: {eval_name}".center(width))
        print("=".center(width, "="))

        try:
            serve_config = _create_serve(args.gpu_id, args.log_level)
            evaluator = load_single_evaluator(eval_name, serve_config)

        except Exception as e:
            logger.exception(e)
            logger.error(f"Failed to initialize evaluator {eval_name}. Skipping...")
            clear_memory()
            continue

        logger.info(f"Hyperparameters: {evaluator.get_hparams()}")

        try:
            eval_results = {}
            for dl, ds_name in zip(loader_list, path_list):
                print()
                print(f"Evaluating on dataset: {ds_name}".center(width))

                results = evaluator.evaluate(dl)
                all_results[ds_name].update(results)
                eval_results[ds_name] = results

        finally:
            evaluator.close()
            clear_memory()

        print()
        for k, v in eval_results.items():
            print(f"Dataset {k}: {v}")

    print()
    print("=".center(width, "="))
    print()
    print("All evaluators tested successfully.")

    for ds_name, results in all_results.items():
        print(f"Results for dataset {ds_name}:\n{results}")
        print()

    for ds_name, dl in zip(path_list, loader_list):
        dl.df.to_csv(ds_name, index=False)
        logger.info(f"Saved results to {ds_name}")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level)
    prepare_environment(seed=args.seed)
    main(args)
