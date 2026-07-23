"""Backend adapters: translate an EngineConfig into a concrete server launch.

Each adapter knows three things about its backend:
  1. how to build the CLI command that starts the server,
  2. how to health-check the OpenAI-compatible HTTP endpoint it exposes,
  3. how to hot-swap LoRA adapters through the backend's management API.
"""

from __future__ import annotations

import abc
import asyncio
import subprocess
import sys
from typing import Any

import httpx

from omniai.engine.config import EngineConfig


class BackendAdapter(abc.ABC):
    """Manages a serving backend as a background subprocess."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.process: subprocess.Popen | None = None

    @abc.abstractmethod
    def build_command(self) -> list[str]:
        """CLI command that launches this backend with the mapped flags."""

    @abc.abstractmethod
    def lora_load_endpoint(self) -> str:
        """Path of the backend's dynamic LoRA-load API."""

    @abc.abstractmethod
    def lora_load_payload(self, name: str, path: str) -> dict[str, Any]:
        """Request body for loading a LoRA adapter."""

    def _extra_flags(self) -> list[str]:
        flags: list[str] = []
        for key, value in self.config.extra_args.items():
            flag = f"--{key.replace('_', '-')}"
            if value is True:
                flags.append(flag)
            else:
                flags.extend([flag, str(value)])
        return flags

    def start(self) -> subprocess.Popen:
        """Launch the backend server in a background process."""
        self.process = subprocess.Popen(
            self.build_command(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self.process

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    async def wait_ready(self, timeout: float = 300.0, interval: float = 2.0) -> bool:
        """Poll the health endpoint until the server answers or timeout."""
        deadline = asyncio.get_event_loop().time() + timeout
        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(f"{self.config.base_url}/health")
                    if resp.status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(interval)
        return False


class VLLMAdapter(BackendAdapter):
    """Adapter for vLLM's OpenAI-compatible server (``vllm serve``)."""

    def build_command(self) -> list[str]:
        cfg = self.config
        cmd = [
            "vllm",
            "serve",
            cfg.model,
            "--host",
            cfg.host,
            "--port",
            str(cfg.port),
        ]
        if cfg.quantization:
            cmd.extend(["--quantization", cfg.quantization])
            if cfg.quantization == "fp8":
                cmd.extend(["--kv-cache-dtype", "fp8"])
        # PagedAttention is vLLM's native KV-cache layout; no flag needed,
        # but reject a request for a cache strategy vLLM cannot provide.
        if cfg.kv_cache not in (None, "paged_attention"):
            raise ValueError(f"vLLM does not support kv_cache={cfg.kv_cache!r}")
        if cfg.tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(cfg.tensor_parallel_size)])
        if cfg.gpu_memory_utilization is not None:
            cmd.extend(["--gpu-memory-utilization", str(cfg.gpu_memory_utilization)])
        if cfg.max_model_len is not None:
            cmd.extend(["--max-model-len", str(cfg.max_model_len)])
        if cfg.enable_lora:
            cmd.extend(["--enable-lora", "--max-loras", "4"])
        cmd.extend(self._extra_flags())
        return cmd

    def lora_load_endpoint(self) -> str:
        return "/v1/load_lora_adapter"

    def lora_load_payload(self, name: str, path: str) -> dict[str, Any]:
        return {"lora_name": name, "lora_path": path}


class SGLangAdapter(BackendAdapter):
    """Adapter for SGLang's server (``python -m sglang.launch_server``)."""

    def build_command(self) -> list[str]:
        cfg = self.config
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            cfg.model,
            "--host",
            cfg.host,
            "--port",
            str(cfg.port),
        ]
        if cfg.quantization:
            cmd.extend(["--quantization", cfg.quantization])
        # RadixAttention (prefix caching) is SGLang's default; only an explicit
        # opt-out disables it.
        if cfg.kv_cache == "disable_radix":
            cmd.append("--disable-radix-cache")
        if cfg.tensor_parallel_size > 1:
            cmd.extend(["--tp", str(cfg.tensor_parallel_size)])
        if cfg.gpu_memory_utilization is not None:
            cmd.extend(["--mem-fraction-static", str(cfg.gpu_memory_utilization)])
        if cfg.max_model_len is not None:
            cmd.extend(["--context-length", str(cfg.max_model_len)])
        if cfg.enable_lora:
            cmd.append("--enable-lora")
        cmd.extend(self._extra_flags())
        return cmd

    def lora_load_endpoint(self) -> str:
        return "/load_lora_adapter"

    def lora_load_payload(self, name: str, path: str) -> dict[str, Any]:
        return {"lora_name": name, "lora_path": path}


ADAPTERS: dict[str, type[BackendAdapter]] = {
    "vllm": VLLMAdapter,
    "sglang": SGLangAdapter,
}
