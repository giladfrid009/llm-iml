from enum import Enum
from src.utils.logging import create_logger
from gserve.configs import ServeConfig, LLMConfig

from src.eval import (
    Evaluator,
    BeaverCost,
    GPTJudge,
    HarmBenchJudge,
    LlamaEvaluator,
    LlamaGuard,
    MDJudge,
    StrongReject,
    WildGuard,
    KeywordMatching,
)


logger = create_logger(__name__)


class EvalName(str, Enum):
    BEAVER = "beaver"
    GPT35_TURBO = "gpt3.5-turbo"
    HARMBENCH = "hb-judge"
    LLAMA2 = "llama2-7b"
    LLAMA3 = "llama3-8b"
    LLAMAGUARD_7B = "llamaguard-7b"
    LLAMAGUARD2_8B = "llamaguard2-8b"
    LLAMAGUARD3_1B = "llamaguard3-1b"
    LLAMAGUARD3_8B = "llamaguard3-8b"
    LLAMAGUARD4_12B = "llamaguard4-12b"
    MDJUDGE = "md-judge"
    STRONGREJECT = "strong-reject"
    WILDGUARD = "wild-guard"
    KEYWORDMATCHING = "keyword-matching"


SUPPORTED_EVALUATORS = [e.value for e in EvalName]


def load_single_evaluator(name: str, serve_config: ServeConfig, **kwargs) -> Evaluator:
    if name == EvalName.BEAVER:
        return BeaverCost(device_map=serve_config.gpu_ids[0], **kwargs)

    if name == EvalName.GPT35_TURBO:
        return GPTJudge(model="gpt-3.5-turbo", max_tokens=5)

    if name == EvalName.HARMBENCH:
        return HarmBenchJudge(serve_config, **kwargs)

    if name == EvalName.LLAMA2:
        return LlamaEvaluator(serve_config, "meta-llama/Llama-2-7b-chat-hf", **kwargs)

    if name == EvalName.LLAMA3:
        return LlamaEvaluator(serve_config, "meta-llama/Llama-3.1-8B-Instruct", **kwargs)

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

    if name == EvalName.MDJUDGE:
        return MDJudge(serve_config, **kwargs)

    if name == EvalName.STRONGREJECT:
        return StrongReject(serve_config, **kwargs)

    if name == EvalName.WILDGUARD:
        return WildGuard(serve_config, **kwargs)

    if name == EvalName.KEYWORDMATCHING:
        return KeywordMatching(**kwargs)

    raise ValueError(f"Unsupported evaluator: {name}.")


def load_evaluators(names: list[str], gpus: int | list[int] = 1) -> list[Evaluator]:
    if isinstance(gpus, int):
        gpus = [gpus]

    evaluators = []

    # special handling of evaluators not requiring a GPU
    NON_GPU = [
        EvalName.GPT35_TURBO.value,
        EvalName.KEYWORDMATCHING.value,
    ]

    for name in names:
        if name in NON_GPU:
            evaluator = load_single_evaluator(name, ServeConfig(gpu_ids=[]))
            evaluators.append(evaluator)

    names = [n for n in names if n not in NON_GPU]

    # now, truncate list of evaluators if there are more evaluators than GPUs
    if len(names) > len(gpus):
        names = names[: len(gpus)]

        logger.warning(
            f"Number of evaluators ({len(names)}) exceeds available GPUs; "
            f"Only following evaluators will be loaded: {names}"
        )

    # for every remaining evaluator assign a matching GPU
    for name, gpu in zip(names, gpus):
        serve_config = ServeConfig(
            gpu_ids=[gpu],
            startup_timeout=20 * 60,
            client_timeout=60,
        )
        evaluator = load_single_evaluator(name, serve_config)
        evaluators.append(evaluator)

    return evaluators
