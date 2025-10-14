import pandas as pd
import sys
import pathlib
import argparse

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from gserve.configs import ServeConfig, LLMConfig
from scripts.utils.load_evaluator import load_single_evaluator, SUPPORTED_EVALUATORS
from src.utils.logging import create_logger
from src.utils.torch import clear_memory
from src.eval import Evaluator
from src.data import TableLoader


logger = create_logger(__name__)


def create_serve(gpu_id: int) -> ServeConfig:
    return ServeConfig(gpu_ids=[gpu_id], startup_timeout=20 * 60, client_timeout=60, verbose=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to evaluation data CSV file.",
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

    return parser.parse_args()


def display_width() -> int:
    try:
        import shutil

        return shutil.get_terminal_size().columns
    except Exception:
        return 120


if __name__ == "__main__":
    args = parse_args()
    data = pd.read_csv(args.path)
    dl_eval = TableLoader(data, batch_size=args.batch_size, shuffle=False)

    width = display_width()  # get terminal width for pretty printing

    all_results = {}
    for eval_name in args.names:
        print()
        print("=".center(width, "="))
        print(f"Running evaluator: {eval_name}".center(width))
        print("=".center(width, "="))

        try:
            evaluator = load_single_evaluator(eval_name, create_serve(args.gpu_id))

        except Exception as e:
            logger.exception(e)
            logger.error(f"Failed to initialize evaluator {eval_name}. Skipping...")
            clear_memory()
            continue

        logger.info(f"Hyperparameters: {evaluator.get_hparams()}")

        try:
            results = evaluator.evaluate(dl_eval)
            all_results.update(results)

        finally:
            evaluator.close()
            clear_memory()

        print()
        print(f"{evaluator.name} results: {results}")

    print()
    print()
    print("All evaluators tested successfully.")
    print("Final results:", all_results)
    data.to_csv(args.path, index=False)
