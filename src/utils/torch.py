import torch
import gc
from pathlib import Path as Path


def get_device() -> torch.device:
    """
    Get the device to use for computations.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_memory() -> None:
    """
    Frees unused memory by calling the garbage collector and clearing the CUDA cache.
    This helps prevent out-of-memory errors in GPU-limited environments.
    """
    gc.collect()
    torch.cuda.empty_cache()


def extract_device(module: torch.nn.Module) -> torch.device:
    """
    Extract the device from a module.
    """
    return next(iter(module.parameters())).device


