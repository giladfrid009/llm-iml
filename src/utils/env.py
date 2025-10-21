import os
import dotenv
import huggingface_hub as hf
import torch
import numpy
import random
from src.utils.logging import create_logger

logger = create_logger(__name__)


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): Seed to set.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_environment(dotenv_path: str | None = None):
    if dotenv_path is None:
        dotenv_path = dotenv.find_dotenv()

    logger.info(f"Using .env file at: {dotenv_path}")
    token_dict = dotenv.dotenv_values(dotenv_path=dotenv_path, verbose=True)

    if not token_dict.get("HF_TOKEN"):
        token_dict["HF_TOKEN"] = hf.get_token()

    token_dict = {k: v for k, v in token_dict.items() if v is not None}  # Filter out empty tokens

    if hf_token := os.getenv("HF_TOKEN"):
        hf.login(token=hf_token)

    os.environ.update(token_dict)
    logger.info(f"Setting API tokens: {list(token_dict.keys())}")

    flag_dict = {
        # "TORCHINDUCTOR_MAX_AUTOTUNE": "1",  # Set to 1 to avoid multithreading issues with vLLM
        # "OMP_NUM_THREADS": "1",  # Set OMP_NUM_THREADS to 1 to avoid multithreading issues with vLLM
        # "NCCL_CUMEM_ENABLE": "0",  # To avoid vLLM bug with NCCL
        # "VLLM_USE_V1": "0",  # Currently ART uses vLLM v0, see art.unsloth.state
        # "VLLM_WORKER_MULTIPROC_METHOD": "spawn",  # To avoid vLLM issues with multiprocessing
        "TOKENIZERS_PARALLELISM": "true",  # Avoid tokenizer parallelism warning
    }

    os.environ.update(flag_dict)
    logger.info(f"Setting additional variables: {flag_dict}")
