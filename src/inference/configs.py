"""Configuration dataclasses for the inference/serving utilities."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vllm.lora.request import LoRARequest


@dataclass
class LLMConfig:
    """Configuration for the underlying vLLM model."""

    model_name: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    quantization: Optional[str] = None
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    seed: int = 0
    gpu_memory_utilization: Optional[float] = None
    enforce_eager: bool = False
    max_model_len: Optional[int] = None
    download_dir: Optional[str] = None
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Optional LoRA adapter to load for all inference calls.
    lora_request: Optional[LoRARequest] = None


@dataclass
class ServeConfig:
    """Configuration for :class:`~src.inference.vllm_service.VLLMService`."""

    gpu_ids: List[int]
    host: str = "127.0.0.1"
    port: Optional[int] = None
    # ``None`` disables the timeout completely
    startup_timeout: Optional[float] = 15.0
    client_timeout: Optional[float] = 30.0
    server_script_path: Optional[str] = None
    verbose: bool = False

