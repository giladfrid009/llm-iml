import torch
from src.utils import clear_memory
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel  # type: ignore
from huggingface_hub import login, HfFolder


def hf_login(hf_token: str | None = None) -> str | None:
    """
    Log in to Hugging Face Hub using the provided token.
    If no token is provided, it will use the token saved in the local cache.

    Args:
        hf_token (str | None): Hugging Face token. If None, it will try to retrieve the token from the local cache.

    Returns:
        str | None: The Hugging Face token if login is successful, otherwise None.
    """
    if hf_token is None:
        hf_token = HfFolder.get_token()

    if hf_token is not None:
        login(token=hf_token)
        HfFolder.save_token(hf_token)

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
        **tokenizer_kwargs,
    )

    if chat_template is not None:
        tokenizer.chat_template = chat_template

    # add pad token if not present
    if not tokenizer.pad_token or not tokenizer.pad_token_id:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[pad]"})

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
        model.load_adapter(
            adapter_name,
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

    return model, tokenizer
