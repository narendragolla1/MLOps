"""Engine configuration and hardware-optimization mapping.

An :class:`EngineConfig` is backend-neutral: users describe *what* they want
(fp8 quantization, paged KV cache, tensor parallelism) and each backend
adapter maps those settings to its own CLI flags.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Backend(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"


class EngineConfig(BaseModel):
    """Backend-neutral serving configuration."""

    model: str
    backend: Backend = Backend.VLLM
    host: str = "127.0.0.1"
    port: int = 8000

    # Hardware optimizations (mapped per-backend by the adapters).
    quantization: str | None = None  # e.g. "fp8", "awq", "gptq"
    kv_cache: str | None = None  # e.g. "paged_attention", "radix_attention"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    enable_lora: bool = True

    # Escape hatch: raw flags appended verbatim to the launch command.
    extra_args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tensor_parallel_size")
    @classmethod
    def _positive_tp(cls, v: int) -> int:
        if v < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        return v

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
