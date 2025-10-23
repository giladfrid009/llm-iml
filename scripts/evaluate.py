import pandas as pd
import sys
import pathlib
import argparse
import random
import torch
import logging
import pprint

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


def _read_df(path: pathlib.Path, **kwargs) -> pd.DataFrame | None:
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

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "data_path",
        type=str,
        nargs="+",
        metavar="PATH",
        help=(
            "Paths to evaluation data file or directory containing data files. "
            "Supported formats: CSV, JSON, JSONL, Excel, TSV, Parquet, Feather, Pickle. "
            "If a directory is provided, all files in the directory will be processed."
        ),
    )

    parser.add_argument(
        "--evaluators",
        type=str,
        nargs="+",
        choices=SUPPORTED_EVALUATORS,
        default=SUPPORTED_EVALUATORS,
        metavar="EVALUATOR",
        help=(
            "List of evaluator names to run. If not provided, all evaluators will be run. "
            f"Available evaluators: {SUPPORTED_EVALUATORS}"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        metavar="N",
        help="Batch size for data loading.",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        metavar="ID",
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

    parser.add_argument(
        "--recurse",
        action="store_true",
        help="Recursively search directories for data files.",
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

    user_dir = pathlib.Path.cwd()
    for path in paths:
        # make all paths relative to CWD
        path = pathlib.Path(path).resolve()
        path = path.relative_to(user_dir, walk_up=True)

        if path.is_dir():
            sub_paths, sub_data = read_data([p.as_posix() for p in path.iterdir() if p.is_file() or args.recurse])
            if len(sub_data) == 0:
                continue

            logger.info(f"Loaded {len(sub_data)} files from directory: {path}")
            path_list.extend(sub_paths)
            data_list.extend(sub_data)
            continue

        try:
            df = _read_df(path)
            if df is None:
                continue

        except Exception as e:
            logger.error(f"Error reading file {path}: {e}. Skipping...")
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
    path_list, data_list = read_data(args.data_path)
    loader_list = [TableLoader(df, batch_size=args.batch_size, shuffle=False) for df in data_list]
    all_results = {ds_name: {} for ds_name in path_list}

    for eval_name in args.evaluators:
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
                print(f"Evaluating: {ds_name}".center(width))

                results = evaluator.evaluate(dl)
                all_results[ds_name].update(results)
                eval_results[ds_name] = results

        finally:
            evaluator.close()
            clear_memory()

        print()
        for k, v in eval_results.items():
            print(f"File {k}:")
            pprint.pprint(v, width=width)

    print()
    print("=".center(width, "="))
    print()

    for ds_name, results in all_results.items():
        print(f"File {ds_name}:")
        pprint.pprint(results, width=width)
        print()

    for ds_name, dl in zip(path_list, loader_list):
        dl.df.to_csv(ds_name, index=False)
        logger.info(f"Saved results to {ds_name}")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level)
    prepare_environment(seed=args.seed)
    main(args)
