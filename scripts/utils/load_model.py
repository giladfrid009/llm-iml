from src.utils.huggingface import load_hf_model
from transformers import PreTrainedModel, PreTrainedTokenizer  # pyright: ignore[reportPrivateImportUsage]
import torch
from src.utils.logging import create_logger

logger = create_logger(__name__)

# from HF https://huggingface.co/lmsys/vicuna-7b-v1.5/discussions/7
VICUNA_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"

# from HF https://huggingface.co/microsoft/Orca-2-13b/commit/e544a7b6a7e1419ea0f61d5e95d0a7ed298be449
ORCA_TEMPLATE = "{{ bos_token }} {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# from https://huggingface.co/mosaicml/mpt-7b-chat/commit/ed874721edb9a72f228b5379b4d488daebb57ed4
MPT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif not 'system' in messages[0]['role'] %}{% set loop_messages = messages %}{% set system_message = 'A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% if system_message != false %}{{ '<|im_start|>system\n' + system_message.strip() + '\n'}}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% else %}{{ '\n' + '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% endif %}{% if (add_generation_prompt == true) %}{{ '\n' + '<|im_start|>' + 'assistant' + '\n' }}{% elif (message['role'] == 'assistant') %}{{ eos_token }}{% endif %}{% endfor %}"

SUPPORTED_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Orenguteng/Llama-3-8B-Lexi-Uncensored",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "lmsys/vicuna-7b-v1.5",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "mosaicml/mpt-7b-chat",
    "microsoft/Orca-2-7b",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-4-mini-instruct",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "openchat/openchat-3.5-0106",
    "HuggingFaceH4/zephyr-7b-beta",
    "google/gemma-2b-it",
    "google/gemma-2-2b-it",
    # "google/gemma-3-1b-it", # NOTE: broken due to HF bug see https://github.com/google-deepmind/gemma/issues/169
    "apple/OpenELM-1_1B-Instruct",
    # robust models
    "GraySwanAI/Llama-3-8B-Instruct-RR",
    "GraySwanAI/Mistral-7B-Instruct-RR",
    "cais/zephyr_7b_r2d2",
    "ContinuousAT/Llama-2-7B-CAT",
    "ContinuousAT/Phi-CAT",
    "ContinuousAT/Phi-CAPO",
    "ContinuousAT/Zephyr-CAT",
    "LLM-LAT/robust-llama3-8b-instruct",
]


def load_model(
    model_name: str,
    torch_dtype: torch.dtype | str = "auto",
    device_map: str = "cuda:0",
    hf_token: str | None = None,
    **kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if kwargs is None:
        kwargs = {}

    config = kwargs.copy()

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' is not supported. Supported models are: {SUPPORTED_MODELS}")

    # Model-Dependent configurations

    if model_name == "lmsys/vicuna-7b-v1.5":
        config.update({"chat_template": VICUNA_TEMPLATE})

    elif model_name == "mosaicml/mpt-7b-chat":
        config.update({"tokenizer_name": "EleutherAI/gpt-neox-20b", "chat_template": MPT_TEMPLATE})

    elif model_name == "microsoft/Orca-2-7b":
        config.update({"chat_template": ORCA_TEMPLATE})

    elif model_name == "ContinuousAT/Llama-2-7B-CAT":
        config.update({"model_name": "meta-llama/Llama-2-7b-chat-hf", "adapter_name": "ContinuousAT/Llama-2-7B-CAT"})

    elif model_name == "ContinuousAT/Phi-CAT":
        config.update({"model_name": "microsoft/Phi-3-mini-4k-instruct", "adapter_name": "ContinuousAT/Phi-CAT"})

    elif model_name == "ContinuousAT/Phi-CAPO":
        config.update({"model_name": "microsoft/Phi-3-mini-4k-instruct", "adapter_name": "ContinuousAT/Phi-CAPO"})

    elif model_name == "ContinuousAT/Zephyr-CAT":
        config.update({"model_name": "HuggingFaceH4/zephyr-7b-beta", "adapter_name": "ContinuousAT/Zephyr-CAT"})

    elif model_name == "apple/OpenELM-1_1B-Instruct":
        config.update({"trust_remote_code": True})

    config.setdefault("model_name", model_name)
    config.setdefault("device_map", device_map)
    config.setdefault("torch_dtype", torch_dtype)
    config.setdefault("hf_token", hf_token)

    return load_hf_model(**config)
