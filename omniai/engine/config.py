"""Engine configuration and hardware-optimization mapping.

An :class:`EngineConfig` is backend-neutral: users describe *what* they want
(fp8 quantization, paged KV cache, tensor parallelism) and each backend
adapter maps those settings to its own CLI flags.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Backend(StrEnum):
    VLLM = "vllm"
    SGLANG = "sglang"


class EngineConfig(BaseModel):
    """Backend-neutral serving configuration."""

    model: str
    backend: Backend = Backend.VLLM
    host: str = "127.0.0.1"
    port: int = 8000

    # Process ownership: managed=True spawns/supervises the server subprocess;
    # managed=False attaches to an already-running server (e.g. the vLLM
    # container in the compose stack) at external_base_url.
    managed: bool = True
    external_base_url: str | None = None

    # Reliability knobs for the HTTP client around the backend.
    request_timeout_s: float = 120.0
    retries: int = 3
    breaker_failure_threshold: int = 5
    breaker_reset_s: float = 30.0
    # Backpressure: bound on concurrent in-flight requests to the backend so
    # a burst that passes the rate limiter cannot pile onto the server.
    max_concurrent_requests: int = 32

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
        if self.external_base_url:
            return self.external_base_url.rstrip("/")
        return f"http://{self.host}:{self.port}"
