"""Tests for backend adapter process lifecycle management."""

import asyncio
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from omniai.engine.backends import BackendAdapter, VLLMAdapter, SGLangAdapter
from omniai.engine.config import EngineConfig


# -- Test fixtures ----------------------------------------------------------


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_test_config(**overrides):
    """Create a test EngineConfig with defaults."""
    defaults = {
        "model": "meta-llama/Llama-2-7b",
        "backend": "vllm",
        "host": "127.0.0.1",
        "port": 8000,
        "devices": None,
        "env": {},
        "extra_args": {},
        "enable_lora": False,
        "max_loras": 4,
        "quantization": None,
        "prefix_caching": True,
        "kv_cache": None,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": None,
        "max_model_len": None,
        "log_dir": None,
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)


# -- Environment building tests ------------------------------------------


def test_build_env_inherits_parent_env():
    """Verify build_env inherits parent environment."""
    os.environ["TEST_VAR"] = "test_value"
    config = create_test_config()
    adapter = VLLMAdapter(config)
    env = adapter.build_env()
    assert env["TEST_VAR"] == "test_value"


def test_build_env_with_single_device():
    """Verify CUDA_VISIBLE_DEVICES is set for single device."""
    config = create_test_config(devices=[0])
    adapter = VLLMAdapter(config)
    env = adapter.build_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_build_env_with_multiple_devices():
    """Verify CUDA_VISIBLE_DEVICES is set for multiple devices."""
    config = create_test_config(devices=[0, 1, 3])
    adapter = VLLMAdapter(config)
    env = adapter.build_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,3"


def test_build_env_no_devices():
    """Verify CUDA_VISIBLE_DEVICES is not set when devices is None."""
    config = create_test_config(devices=None)
    adapter = VLLMAdapter(config)
    env = adapter.build_env()
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_build_env_empty_devices():
    """Verify CUDA_VISIBLE_DEVICES handles empty device list."""
    config = create_test_config(devices=[])
    adapter = VLLMAdapter(config)
    env = adapter.build_env()
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_build_env_backend_specific_vars():
    """Verify backend-specific env vars are included."""
    config = create_test_config(enable_lora=True)
    adapter = VLLMAdapter(config)
    env = adapter.build_env()
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"


def test_build_env_user_override_priority():
    """Verify user env overrides take priority."""
    config = create_test_config(
        devices=[0],
        enable_lora=True,
        env={"CUDA_VISIBLE_DEVICES": "2,3", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "False"},
    )
    adapter = VLLMAdapter(config)
    env = adapter.build_env()
    # User overrides should win
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "False"


def test_build_env_custom_user_vars():
    """Verify custom user env vars are included."""
    config = create_test_config(env={"CUSTOM_VAR": "custom_value", "ANOTHER": "value"})
    adapter = VLLMAdapter(config)
    env = adapter.build_env()
    assert env["CUSTOM_VAR"] == "custom_value"
    assert env["ANOTHER"] == "value"


# -- Extra flags tests ---------------------------------------------------


def test_extra_flags_boolean_flags():
    """Verify boolean extra_args are added as flags without values."""
    config = create_test_config(extra_args={"enable_foo": True, "disable_bar": True})
    adapter = VLLMAdapter(config)
    flags = adapter._extra_flags()
    assert "--enable-foo" in flags
    assert "--disable-bar" in flags
    # Should not have values after boolean flags
    foo_idx = flags.index("--enable-foo")
    assert foo_idx == len(flags) - 1 or flags[foo_idx + 1].startswith("--")


def test_extra_flags_value_flags():
    """Verify value extra_args are added as flag-value pairs."""
    config = create_test_config(extra_args={"cpu_offload_gb": 4, "num_workers": 8})
    adapter = VLLMAdapter(config)
    flags = adapter._extra_flags()
    assert "--cpu-offload-gb" in flags
    assert "4" in flags
    assert "--num-workers" in flags
    assert "8" in flags


def test_extra_flags_underscores_to_dashes():
    """Verify underscores in flag names are converted to dashes."""
    config = create_test_config(extra_args={"enable_cuda_graph": True})
    adapter = VLLMAdapter(config)
    flags = adapter._extra_flags()
    assert "--enable-cuda-graph" in flags
    assert "--enable_cuda_graph" not in flags


def test_extra_flags_mixed():
    """Verify mixed boolean and value flags work together."""
    config = create_test_config(
        extra_args={
            "enable_prefill": True,
            "num_gpu_blocks": 1024,
            "trust_remote_code": True,
        }
    )
    adapter = VLLMAdapter(config)
    flags = adapter._extra_flags()
    assert "--enable-prefill" in flags
    assert "--trust-remote-code" in flags
    assert "--num-gpu-blocks" in flags
    assert "1024" in flags


def test_extra_flags_empty():
    """Verify empty extra_args returns empty flags list."""
    config = create_test_config(extra_args={})
    adapter = VLLMAdapter(config)
    flags = adapter._extra_flags()
    assert flags == []


# -- Log file handling tests ------------------------------------------


def test_open_log_with_none_log_dir():
    """Verify log returns DEVNULL when log_dir is None."""
    config = create_test_config(log_dir=None)
    adapter = VLLMAdapter(config)
    sink = adapter._open_log()
    assert sink is subprocess.DEVNULL


def test_open_log_creates_directory():
    """Verify log directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "new" / "nested" / "dir"
        config = create_test_config(log_dir=str(log_dir))
        adapter = VLLMAdapter(config)
        sink = adapter._open_log()
        assert log_dir.exists()
        sink.close()


def test_open_log_file_naming():
    """Verify log file is named correctly with backend and port."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = create_test_config(log_dir=tmpdir, backend="vllm", port=8000)
        adapter = VLLMAdapter(config)
        sink = adapter._open_log()
        expected_name = "vllm-8000.log"
        assert (Path(tmpdir) / expected_name).exists()
        sink.close()


def test_open_log_append_mode():
    """Verify log file is opened in append mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = create_test_config(log_dir=tmpdir)
        adapter = VLLMAdapter(config)

        # First call
        sink1 = adapter._open_log()
        sink1.write(b"first write\n")
        sink1.close()

        # Second call should append
        adapter2 = VLLMAdapter(config)
        sink2 = adapter2._open_log()
        sink2.write(b"second write\n")
        sink2.close()

        log_file = list(Path(tmpdir).glob("*.log"))[0]
        content = log_file.read_bytes()
        assert b"first write" in content
        assert b"second write" in content


def test_open_log_file_handle_stored():
    """Verify log file handle is stored for later closing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = create_test_config(log_dir=tmpdir)
        adapter = VLLMAdapter(config)
        sink = adapter._open_log()
        assert adapter._log_file is not None
        assert not adapter._log_file.closed
        adapter._log_file.close()


# -- Process lifecycle tests ------------------------------------------


def test_process_initially_none():
    """Verify process is None before start() is called."""
    config = create_test_config()
    adapter = VLLMAdapter(config)
    assert adapter.process is None


def test_is_alive_when_no_process():
    """Verify is_alive returns False when process is None."""
    config = create_test_config()
    adapter = VLLMAdapter(config)
    assert not adapter.is_alive()


def test_is_alive_checks_poll():
    """Verify is_alive checks process.poll()."""
    config = create_test_config()
    adapter = VLLMAdapter(config)

    # Mock a running process
    mock_process = mock.MagicMock()
    mock_process.poll.return_value = None  # None means still running
    adapter.process = mock_process
    assert adapter.is_alive()

    # Mock an exited process
    mock_process.poll.return_value = 1  # Non-None means exited
    assert not adapter.is_alive()


def test_stop_when_no_process():
    """Verify stop() is safe when no process exists."""
    config = create_test_config()
    adapter = VLLMAdapter(config)
    # Should not raise
    adapter.stop()
    assert adapter.process is None


@pytest.mark.asyncio
async def test_wait_ready_success():
    """Verify wait_ready returns True on successful health check."""
    config = create_test_config()
    adapter = VLLMAdapter(config)

    async def mock_handler(request):
        if "health" in request.url.path:
            return mock.MagicMock(status_code=200)
        raise Exception("unexpected endpoint")

    with mock.patch("omniai.engine.backends.httpx.AsyncClient") as MockClient:
        mock_client = mock.AsyncMock()
        mock_client.get = mock.AsyncMock(return_value=mock.MagicMock(status_code=200))
        MockClient.return_value.__aenter__.return_value = mock_client

        result = await adapter.wait_ready(timeout=1.0, interval=0.01)
        assert result is True


@pytest.mark.asyncio
async def test_wait_ready_timeout():
    """Verify wait_ready returns False on timeout."""
    config = create_test_config()
    adapter = VLLMAdapter(config)

    with mock.patch("omniai.engine.backends.httpx.AsyncClient") as MockClient:
        mock_client = mock.AsyncMock()
        # Always return 503 (service unavailable)
        mock_client.get = mock.AsyncMock(return_value=mock.MagicMock(status_code=503))
        MockClient.return_value.__aenter__.return_value = mock_client

        result = await adapter.wait_ready(timeout=0.05, interval=0.01)
        assert result is False


@pytest.mark.asyncio
async def test_wait_ready_retries_on_error():
    """Verify wait_ready retries on httpx errors."""
    config = create_test_config()
    adapter = VLLMAdapter(config)

    call_count = {"n": 0}

    async def failing_get(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise Exception("connection refused")
        return mock.MagicMock(status_code=200)

    with mock.patch("omniai.engine.backends.httpx.AsyncClient") as MockClient:
        mock_client = mock.AsyncMock()
        mock_client.get = failing_get
        MockClient.return_value.__aenter__.return_value = mock_client

        result = await adapter.wait_ready(timeout=1.0, interval=0.01)
        assert result is True
        assert call_count["n"] >= 2


# -- VLLMAdapter specific tests -------------------------------------------


def test_vllm_adapter_build_command_basic():
    """Verify VLLMAdapter builds correct basic command."""
    config = create_test_config()
    adapter = VLLMAdapter(config)
    cmd = adapter.build_command()
    assert cmd[0] == "vllm"
    assert cmd[1] == "serve"
    assert "meta-llama/Llama-2-7b" in cmd
    assert "--host" in cmd
    assert "127.0.0.1" in cmd
    assert "--port" in cmd
    assert "8000" in cmd


def test_vllm_adapter_quantization():
    """Verify VLLMAdapter includes quantization flags."""
    config = create_test_config(quantization="fp8")
    adapter = VLLMAdapter(config)
    cmd = adapter.build_command()
    assert "--quantization" in cmd
    assert "fp8" in cmd
    assert "--kv-cache-dtype" in cmd
    assert "fp8" in cmd


def test_vllm_adapter_prefix_caching():
    """Verify VLLMAdapter includes prefix caching flag."""
    config = create_test_config(prefix_caching=True)
    adapter = VLLMAdapter(config)
    cmd = adapter.build_command()
    assert "--enable-prefix-caching" in cmd


def test_vllm_adapter_tensor_parallel():
    """Verify VLLMAdapter includes tensor parallel size."""
    config = create_test_config(tensor_parallel_size=4)
    adapter = VLLMAdapter(config)
    cmd = adapter.build_command()
    assert "--tensor-parallel-size" in cmd
    assert "4" in cmd


def test_vllm_adapter_lora_flags():
    """Verify VLLMAdapter sets LoRA flags when enabled."""
    config = create_test_config(enable_lora=True, max_loras=8)
    adapter = VLLMAdapter(config)
    cmd = adapter.build_command()
    assert "--enable-lora" in cmd
    assert "--max-loras" in cmd
    assert "8" in cmd


def test_vllm_adapter_lora_endpoints():
    """Verify VLLMAdapter exposes correct LoRA endpoints."""
    config = create_test_config()
    adapter = VLLMAdapter(config)
    assert adapter.lora_load_endpoint() == "/v1/load_lora_adapter"
    assert adapter.lora_unload_endpoint() == "/v1/unload_lora_adapter"


def test_vllm_adapter_lora_payload():
    """Verify VLLMAdapter creates correct LoRA payloads."""
    config = create_test_config()
    adapter = VLLMAdapter(config)
    load_payload = adapter.lora_load_payload("adapter-v2", "/path/to/adapter")
    assert load_payload == {"lora_name": "adapter-v2", "lora_path": "/path/to/adapter"}

    unload_payload = adapter.lora_unload_payload("adapter-v2")
    assert unload_payload == {"lora_name": "adapter-v2"}


# -- SGLangAdapter specific tests ------------------------------------------


def test_sglang_adapter_build_command_basic():
    """Verify SGLangAdapter builds correct basic command."""
    config = create_test_config(backend="sglang")
    adapter = SGLangAdapter(config)
    cmd = adapter.build_command()
    assert "sglang.launch_server" in cmd
    assert "--model-path" in cmd
    assert "meta-llama/Llama-2-7b" in cmd


def test_sglang_adapter_prefix_caching_disabled():
    """Verify SGLangAdapter disables radix cache when prefix caching is off."""
    config = create_test_config(backend="sglang", prefix_caching=False)
    adapter = SGLangAdapter(config)
    cmd = adapter.build_command()
    assert "--disable-radix-cache" in cmd


def test_sglang_adapter_quantization():
    """Verify SGLangAdapter includes quantization."""
    config = create_test_config(backend="sglang", quantization="int8")
    adapter = SGLangAdapter(config)
    cmd = adapter.build_command()
    assert "--quantization" in cmd
    assert "int8" in cmd


def test_sglang_adapter_lora_endpoints():
    """Verify SGLangAdapter exposes correct LoRA endpoints."""
    config = create_test_config(backend="sglang")
    adapter = SGLangAdapter(config)
    assert adapter.lora_load_endpoint() == "/load_lora_adapter"
    assert adapter.lora_unload_endpoint() == "/unload_lora_adapter"


# -- Signal handling tests -----------------------------------------------


def test_signal_group_with_killpg():
    """Verify _signal_group uses killpg on POSIX systems."""
    config = create_test_config()
    adapter = VLLMAdapter(config)

    mock_process = mock.MagicMock()
    mock_process.pid = 1234
    adapter.process = mock_process

    with mock.patch("os.killpg") as mock_killpg:
        with mock.patch("os.getpgid", return_value=1234):
            adapter._signal_group(signal.SIGTERM)
            mock_killpg.assert_called_once_with(1234, signal.SIGTERM)


def test_signal_group_handles_process_lookup_error():
    """Verify _signal_group handles ProcessLookupError gracefully."""
    config = create_test_config()
    adapter = VLLMAdapter(config)

    mock_process = mock.MagicMock()
    mock_process.pid = 9999
    adapter.process = mock_process

    with mock.patch("os.killpg", side_effect=ProcessLookupError):
        # Should not raise
        adapter._signal_group(signal.SIGTERM)


def test_stop_sends_sigterm_then_sigkill():
    """Verify stop() sends SIGTERM first, then SIGKILL if timeout."""
    config = create_test_config()
    adapter = VLLMAdapter(config)

    mock_process = mock.MagicMock()
    mock_process.poll.return_value = 1  # Already exited
    adapter.process = mock_process

    with mock.patch.object(adapter, "_signal_group") as mock_signal:
        adapter.stop()
        mock_signal.assert_called_with(signal.SIGTERM)


def test_stop_closes_log_file():
    """Verify stop() closes the log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = create_test_config(log_dir=tmpdir)
        adapter = VLLMAdapter(config)
        sink = adapter._open_log()

        assert adapter._log_file is not None
        assert not adapter._log_file.closed

        adapter.stop()
        assert adapter._log_file is None
