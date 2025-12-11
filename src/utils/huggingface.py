import torch
import os
from src.utils.torch import clear_memory
from src.utils.logging import create_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel  # type: ignore
import huggingface_hub


logger = create_logger(__name__)


def hf_login(hf_token: str | None = None) -> str | None:
    """
    Log in to Hugging Face Hub using the provided token.
    If no token is provided, it will try to retrieve the token from the local cache or environment variable.

    Args:
        hf_token (str | None): Hugging Face token. If None, it will try to retrieve the token from the local cache.

    Returns:
        str | None: The Hugging Face token if login is successful, otherwise None.
    """
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")

    if hf_token is None:
        hf_token = huggingface_hub.get_token()

    if hf_token is not None:
        huggingface_hub.login(token=hf_token)

    else:
        logger.warning("No Hugging Face token provided or found. Skipping login to Hugging Face Hub.")

    return hf_token


def load_hf_tokenizer(
    tokenizer_name: str,
    chat_template: str | None = None,
    tokenizer_kwargs: dict | None = None,
    trust_remote_code: bool = False,
    hf_token: str | None = None,
) -> PreTrainedTokenizer:
    hf_login(hf_token)

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        **tokenizer_kwargs,
    )

    if chat_template is not None:
        tokenizer.chat_template = chat_template

    # add pad token if not present
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            logger.info(f"Set pad_token to unk_token (id: {tokenizer.pad_token_id})")
        elif tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token (id: {tokenizer.pad_token_id})")
        else:
            tokenizer.add_special_tokens({"pad_token": "[pad]"})
            logger.info(f"Added new pad_token '[pad]' (id: {tokenizer.pad_token_id})")

    return tokenizer


def load_hf_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda:0",
    tokenizer_name: str | None = None,
    adapter_name: str | None = None,
    trust_remote_code: bool = False,
    *,
    chat_template: str | None = None,
    model_kwargs: dict | None = None,
    tokenizer_kwargs: dict | None = None,
    adapter_kwargs: dict | None = None,
    hf_token: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if model_kwargs is None:
        model_kwargs = {}

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    if adapter_kwargs is None:
        adapter_kwargs = {}

    hf_login(hf_token)
    clear_memory()

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )

    if adapter_name is not None:
        logger.debug(f"Loading adapter: {adapter_name}")
        model.load_adapter(
            peft_model_id=adapter_name,
            device_map=device_map,
            **adapter_kwargs,
        )

    if tokenizer_name is None:
        tokenizer_name = model_name

    tokenizer = load_hf_tokenizer(
        tokenizer_name,
        chat_template=chat_template,
        trust_remote_code=trust_remote_code,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
