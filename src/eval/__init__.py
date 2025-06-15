from src.eval.evaluator import Evaluator, MultiEvaluator

from src.eval.beaver_evaluator import BeaverEvaluator
from src.eval.harmbench_evaluator import HarmBenchEvaluator
from src.eval.llama_evaluator import LlamaEvaluator
from src.eval.llama_guard_evaluator import LlamaGuardEvaluator
from src.eval.strong_reject_evaluator import StrongRejectEvaluator
from src.eval.template_evaluator import TemplateEvaluator

# TODO: GO OVER ALL EVALUATORS AND MAKE SURE THAT WE ACTUALLY CALL IT CORRECTLY
# IT DOESNT MAKE SENSE TO CALL A CHAT EVALUATOR VIA GENERATION

# TODO: test whether loading the model in float16 and bfloat16 influences the results
# also, check it for the rest of the codebase

__all__ = [
    "Evaluator",
    "MultiEvaluator",
    "BeaverEvaluator",
    "HarmBenchEvaluator",
    "LlamaEvaluator",
    "LlamaGuardEvaluator",
    "StrongRejectEvaluator",
    "TemplateEvaluator",
]