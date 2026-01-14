from enum import Enum
from src.utils.logging import create_logger
from src.utils.torch import clear_memory
from gserve.configs import ServeConfig, LLMConfig

from src.eval import (
    Evaluator,
    BeaverCost,
    HarmBenchJudge,
    LlamaEvaluator,
    LlamaGuard,
    MDJudge,
    JBBJudge,
    StrongReject,
    WildGuard,
    KeywordMatching,
)


logger = create_logger(__name__)


class EvalName(str, Enum):
    BEAVER = "beaver"
    HARMBENCH = "hb-judge"
    LLAMA2 = "llama2-7b"
    # LLAMA3 = "llama3-8b" # NOTE: currently broken
    LLAMAGUARD_7B = "llamaguard-7b"
    LLAMAGUARD2_8B = "llamaguard2-8b"
    LLAMAGUARD3_1B = "llamaguard3-1b"
    LLAMAGUARD3_8B = "llamaguard3-8b"
    LLAMAGUARD4_12B = "llamaguard4-12b"
    # MDJUDGE = "md-judge" # NOTE: currently broken
    JBBJUDGE = "jbb-judge"
    STRONGREJECT = "strong-reject"
    WILDGUARD = "wild-guard"
    KEYWORDMATCHING = "keyword-matching"


SUPPORTED_EVALUATORS = [e.value for e in EvalName]


def load_single_evaluator(name: str, serve_config: ServeConfig, **kwargs) -> Evaluator:
    if name == EvalName.BEAVER:
        return BeaverCost(device_map=serve_config.gpu_ids[0], **kwargs)

    if name == EvalName.HARMBENCH:
        return HarmBenchJudge(serve_config, **kwargs)

    if name == EvalName.LLAMA2:
        return LlamaEvaluator(serve_config, "meta-llama/Llama-2-7b-chat-hf", **kwargs)

    # if name == EvalName.LLAMA3: # NOTE: currently broken
    #     return LlamaEvaluator(serve_config, "meta-llama/Llama-3.1-8B-Instruct", **kwargs)

    if name == EvalName.LLAMAGUARD_7B:
        return LlamaGuard(serve_config, "meta-llama/LlamaGuard-7b", **kwargs)

    if name == EvalName.LLAMAGUARD2_8B:
        return LlamaGuard(serve_config, "meta-llama/Meta-Llama-Guard-2-8B", **kwargs)

    if name == EvalName.LLAMAGUARD3_1B:
        return LlamaGuard(serve_config, "meta-llama/Llama-Guard-3-1B", **kwargs)

    if name == EvalName.LLAMAGUARD3_8B:
        return LlamaGuard(serve_config, "meta-llama/Llama-Guard-3-8B", **kwargs)

    if name == EvalName.LLAMAGUARD4_12B:
        return LlamaGuard(serve_config, "meta-llama/Llama-Guard-4-12B", **kwargs)

    # if name == EvalName.MDJUDGE: # NOTE: currently broken
    #     return MDJudge(serve_config, **kwargs)

    if name == EvalName.JBBJUDGE:
        return JBBJudge(serve_config, **kwargs)

    if name == EvalName.STRONGREJECT:
        return StrongReject(serve_config, **kwargs)

    if name == EvalName.WILDGUARD:
        return WildGuard(serve_config, **kwargs)

    if name == EvalName.KEYWORDMATCHING:
        return KeywordMatching(**kwargs)

    raise ValueError(f"Unsupported evaluator: {name}.")


def _align_gpus(names: list[str], gpus: list[int]) -> list[int]:
    CPU_EVALS = [
        EvalName.KEYWORDMATCHING.value,
    ]

    available_gpus = gpus.copy()
    aligned_gpus = []

    for eval_name in names:
        if eval_name in CPU_EVALS:
            aligned_gpus.append(-1)

        else:
            if not available_gpus:
                raise ValueError(
                    f"Not enough GPUs ({gpus}) for the requested evaluators ({names}); Please reduce the number of evaluators or add more GPUs."
                )

            gpu_id = available_gpus.pop(0)
            aligned_gpus.append(gpu_id)

    return aligned_gpus


def load_evaluators(names: list[str], gpus: int | list[int] = 1) -> list[Evaluator]:
    if isinstance(gpus, int):
        gpus = [gpus]

    # adds fictitious GPU (-1) for non-GPU evaluators
    gpus = _align_gpus(names, gpus)

    clear_memory()

    evaluators = []
    for name, gpu in zip(names, gpus):
        serve_config = ServeConfig(
            gpu_ids=[gpu],
            startup_timeout=20 * 60,
            client_timeout=2 * 60,
        )
        evaluator = load_single_evaluator(name, serve_config)
        evaluators.append(evaluator)

    return evaluators
