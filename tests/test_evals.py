import pandas as pd
from typing import Generator
import sys
import pathlib
from enum import Enum
import argparse

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.utils.torch import clear_memory
from src.eval import Evaluator, MultiEvaluator
from src.data import TableLoader

from src.eval import (
    BeaverCost,
    HarmBenchJudge,
    LlamaEvaluator,
    LlamaGuard,
    MDJudge,
    StrongReject,
    WildGuard,
    KeywordMatching,
)

from gserve.configs import ServeConfig, LLMConfig


class EvalNames(str, Enum):
    BEAVERCOST = "BeaverCost"
    HARMBENCHJUDGE = "HarmBenchJudge"
    LLAMAEVALUATOR2_7B = "LlamaEvaluator-2-7b"
    LLAMAEVALUATOR3_8B = "LlamaEvaluator-3-8b"
    LLAMAGUARD_7B = "LlamaGuard-7b"
    LLAMAGUARD2_8B = "LlamaGuard-2-8b"
    LLAMAGUARD3_1B = "LlamaGuard-3-1b"
    LLAMAGUARD3_8B = "LlamaGuard-3-8b"
    LLAMAGUARD4_12B = "LlamaGuard-4-12b"
    MDJUDGE = "MDJudge"
    STRONGREJECT = "StrongReject"
    WILDGUARD = "WildGuard"
    KEYWORDMATCHING = "KeywordMatching"
    MULTI_EVAL = "MultiEvaluator"


def yield_evaluators(names: list[str] | None) -> Generator[Evaluator, None, None]:
    serve_config = ServeConfig(
        gpu_ids=[0],
        startup_timeout=20 * 60,
        client_timeout=60,
        verbose=True,
    )

    if names is None or EvalNames.BEAVERCOST.value in names:
        yield BeaverCost(device_map="cuda:0")

    if names is None or EvalNames.HARMBENCHJUDGE.value in names:
        yield HarmBenchJudge(serve_config)

    if names is None or EvalNames.LLAMAEVALUATOR2_7B.value in names:
        yield LlamaEvaluator(serve_config, model_name="meta-llama/Llama-2-7b-chat-hf")

    if names is None or EvalNames.LLAMAEVALUATOR3_8B.value in names:
        yield LlamaEvaluator(serve_config, model_name="meta-llama/Llama-3.1-8B-Instruct")

    if names is None or EvalNames.LLAMAGUARD_7B.value in names:
        yield LlamaGuard(serve_config, model_name="meta-llama/LlamaGuard-7b")

    if names is None or EvalNames.LLAMAGUARD2_8B.value in names:
        yield LlamaGuard(serve_config, model_name="meta-llama/Meta-Llama-Guard-2-8B")

    if names is None or EvalNames.LLAMAGUARD3_1B.value in names:
        yield LlamaGuard(serve_config, model_name="meta-llama/Llama-Guard-3-1B")

    if names is None or EvalNames.LLAMAGUARD3_8B.value in names:
        yield LlamaGuard(serve_config, model_name="meta-llama/Llama-Guard-3-8B")

    if names is None or EvalNames.LLAMAGUARD4_12B.value in names:
        yield LlamaGuard(serve_config, model_name="meta-llama/Llama-Guard-4-12B")

    if names is None or EvalNames.MDJUDGE.value in names:
        yield MDJudge(serve_config)

    if names is None or EvalNames.STRONGREJECT.value in names:
        yield StrongReject(serve_config)

    if names is None or EvalNames.WILDGUARD.value in names:
        yield WildGuard(serve_config)

    if names is None or EvalNames.KEYWORDMATCHING.value in names:
        yield KeywordMatching()

    if names is None or EvalNames.MULTI_EVAL.value in names:
        yield MultiEvaluator(
            evaluators=[KeywordMatching(), KeywordMatching()],
            combine_fn=lambda res: res["Matching/GCG2"] + res["Matching/GCG1"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_name",
        type=str,
        nargs="*",
        default=None,
        help="List of evaluator names to run. If not provided, all evaluators will be run.",
        choices=[e.value for e in EvalNames],
    )

    args = parser.parse_args()
    data = pd.read_csv("tests/eval_data.csv")
    dl_eval = TableLoader(data, batch_size=200, shuffle=False)

    eval_results = {}

    for ev in yield_evaluators(args.eval_name):
        print()
        print("=".center(80, "="))
        print(f"Testing evaluator: {ev.name}")
        print(ev.get_hparams())
        print("=".center(80, "="))

        results = ev.evaluate(dl_eval)
        eval_results.update(results)

        print("=".center(80, "="))
        print(f"{ev.name} Results: {results}")
        print("=".center(80, "="))

        ev.close()
        clear_memory()

    print("All evaluators tested successfully.")
    print("Final results:", eval_results)
