# fix imports
import os
import sys

module_path = os.path.abspath(os.path.join("."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.eval.harmbench_evaluator import HarmbenchEvaluator
from src.eval.llama_evaluator import LlamaEvaluator
from src.eval.template_evaluator import TemplateEvaluator
from src.eval.llama_guard_evaluator import LlamaGuardEvaluator
from src.eval.strong_reject_evaluator import StrongRejectEvaluator
from src.eval.beaver_evaluator import BeaverEvaluator
from gserve.configs import ServeConfig, LLMConfig

import torch
import pandas as pd
from src.data import DF_Batcher
import time

if __name__ == "__main__":

    evaluators = [
        # HarmbenchEvaluator(
        #     serve_config=ServeConfig(gpu_ids=[1], startup_timeout=5 * 60, client_timeout=60),
        #     use_context=False,
        #     silent=False,
        # ),
        # LlamaGuardEvaluator(
        #     serve_config=ServeConfig(gpu_ids=[1], startup_timeout=None, client_timeout=60, verbose=True),
        #     # llm_config=LLMConfig(model_name="meta-llama/Llama-Guard-4-12B", max_model_len=4096),
        #     model_name="meta-llama/Llama-Guard-3-1B",
        #     silent=False,
        # ),
        # StrongRejectEvaluator(
        #     serve_config=ServeConfig(gpu_ids=[1], startup_timeout=5 * 60, client_timeout=60),
        #     binary_thresh=0.5,
        #     silent=False,
        # ),
        BeaverEvaluator(
            device_map="cuda:1",
            binary_thresh=None,
            silent=False,
            compile=True,
        ),
        # LlamaEvaluator(
        #     serve_config=ServeConfig(gpu_ids=[1], startup_timeout=5 * 60, client_timeout=60),
        #     silent=False,
        # ),
        # TemplateEvaluator(
        #     silent=False,
        # ),
    ]

    ds_eval = pd.read_csv("tests/test_responses.csv")
    dl_eval = DF_Batcher(ds_eval, batch_size=50, shuffle=False)

    for i in range(5):

        torch.cuda.empty_cache()
        start = time.time()
        eval_results = []

        for evaluator in evaluators:
            print(f"Running evaluator: {evaluator.name}")
            results = evaluator.evaluate(dl_eval)
            eval_results.append(results)
            print(f"Results: {results}")

        end = time.time()
        print(f"Eval {i} - Time {end - start:.2f}")
