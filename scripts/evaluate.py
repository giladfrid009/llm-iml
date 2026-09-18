import pandas as pd
import sys
import pathlib
import argparse
import random
import torch
import logging
import pprint
import fnmatch
from tqdm.auto import tqdm

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from gserve.configs import ServeConfig
from scripts.utils.load_evaluator import load_single_evaluator, SUPPORTED_EVALUATORS
from src.utils.logging import create_logger, setup_logging, loglevel_names, parse_log_level
from src.utils.torch import clear_memory
from src.eval import Evaluator
from src.data import TableLoader
from src.utils import env


logger = create_logger(__name__)


def _create_serve(gpu_id: int, loglevel: str) -> ServeConfig:
    verbose = parse_log_level(loglevel) <= logging.DEBUG
    return ServeConfig(gpu_ids=[gpu_id], startup_timeout=20 * 60, client_timeout=2 * 60, verbose=verbose)


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
        help=(f"List of evaluator names to run. If not provided, all evaluators will be run. Available evaluators: {SUPPORTED_EVALUATORS}"),
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

    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=["*"],
        metavar="PATTERN",
        help="List of filename patterns to include when searching directories.",
    )

    args = parser.parse_args()

    # print the parsed arguments
    print()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    return args


def read_data(paths: list[str], recurse: bool, patterns: list[str]) -> tuple[list[str], list[pd.DataFrame]]:
    path_list = []
    data_list = []

    def matches_patterns(file_path: pathlib.Path) -> bool:
        """Check if file path matches any of the patterns."""
        # Use forward-slash normalized path for consistent pattern matching
        path_str = file_path.as_posix()
        return any(fnmatch.fnmatch(path_str, pattern) for pattern in patterns)

    user_dir = pathlib.Path.cwd()
    for path in paths:
        # make all paths relative to CWD
        path = pathlib.Path(path).resolve()
        path = path.relative_to(user_dir, walk_up=True)

        if path.is_dir():
            inner_paths = [p.as_posix() for p in path.iterdir() if (p.is_file() and matches_patterns(p)) or (p.is_dir() and recurse)]
            sub_paths, sub_data = read_data(inner_paths, recurse, patterns)
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
            continue

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


def _create_evaluator(eval_name: str, serve_config: ServeConfig) -> Evaluator | None:
    try:
        clear_memory()
        evaluator = load_single_evaluator(eval_name, serve_config)
        return evaluator

    except Exception as e:
        logger.exception(e)
        clear_memory()
        return None


def main(args: argparse.Namespace):
    # get terminal width for pretty printing
    width = _display_width()

    # read data
    path_list, data_list = read_data(args.data_path, args.recurse, args.patterns)
    loader_list = [TableLoader(df, batch_size=args.batch_size, shuffle=False) for df in data_list]
    all_results = {ds_name: {} for ds_name in path_list}
    had_failures = False

    if len(loader_list) == 0:
        logger.error("No valid data files found. Exiting.")
        return

    for eval_name in args.evaluators:
        print()
        print("=".center(width, "="))
        print(f"Running evaluator: {eval_name}".center(width))
        print("=".center(width, "="))

        serve_config = _create_serve(args.gpu_id, args.log_level)
        evaluator = _create_evaluator(eval_name, serve_config)
        if evaluator is None:
            logger.error("Skipping evaluator due to initialization failure.")
            had_failures = True
            continue

        logger.info(f"Hyperparameters: {evaluator.get_hparams()}")

        eval_results = {}
        for dl, ds_name in tqdm(zip(loader_list, path_list), total=len(loader_list), desc="Files", leave=False):
            if evaluator is None:
                logger.info("Re-initializing evaluator...")
                evaluator = _create_evaluator(eval_name, serve_config)
                if evaluator is None:
                    logger.error("Failed to re-initialize evaluator. Skipping remaining datasets.")
                    had_failures = True
                    break

            try:
                results = evaluator.evaluate(dl)
                all_results[ds_name].update(results)
                eval_results[ds_name] = results
                
                # save results
                output_path = str(pathlib.Path(ds_name).with_suffix(".csv"))
                output = pathlib.Path(output_path)
                temporary_output = output.with_name(f".{output.name}.tmp")
                dl.df.to_csv(temporary_output, index=False)
                temporary_output.replace(output)
                logger.info(f"Saved results to {output_path}")

            except Exception as e:
                logger.exception(e)
                logger.error(f"Evaluation failed on dataset {ds_name}.")
                had_failures = True
                evaluator.close()
                evaluator = None

        if evaluator is not None:
            evaluator.close()

        print()
        for k, v in eval_results.items():
            print(f"File {k}:")
            pprint.pprint(v, width=width)

    print()
    print("=".center(width, "="))
    print("Overall Results".center(width))
    print("=".center(width, "="))

    for ds_name, results in all_results.items():
        print(f"File {ds_name}:")
        pprint.pprint(results, width=width)
        print()

    if had_failures:
        raise RuntimeError("One or more evaluator jobs failed; see errors above")

if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level)
    prepare_environment(seed=args.seed)
    main(args)
