import torch
import numpy
import random
import gc
from pathlib import Path as Path


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): Seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)


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


def sample_lp_ball(length: int, norm: float = 2.0, device: torch.device | None = None) -> torch.Tensor:
    """
    Implementation of the method described in:
    [https://stats.stackexchange.com/questions/352668/generate-uniform-noise-from-a-p-norm-ball-x-p-leq-r]

    Sample a random vector from the Lp ball of radius 1.

    Args:
        length (int): Length of the vector.
        norm (int): Norm of the ball.
        device (torch.device, optional): Device to move the tensor to.

    Returns:
        torch.Tensor: A random vector sampled from the Lp ball.
    """
    if device is None:
        device = torch.device("cpu")
    vec = (-torch.log(torch.rand(length, device=device))) ** (1 / norm)
    sgn = 2 * torch.randint(0, 2, (length,), dtype=torch.float32, device=device) - 1
    vec = sgn * vec
    vec = vec / (torch.norm(vec, p=norm) + torch.finfo(vec.dtype).eps)
    rad = torch.exp(torch.log(torch.rand(1, device=device)) / length)
    return rad * vec


def api_key_from_file(path: str) -> str:
    """
    Read an API key from a file.

    Args:
        path (str): Path to the file containing the API key.

    Returns:
        str: The API key.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    key_file = Path(path)
    if key_file.exists():
        with key_file.open("r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        raise FileNotFoundError("API key file not found")
