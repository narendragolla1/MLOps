"""SandboxExecution: run LLM-generated code in a disposable Docker container.

Each execution launches a fresh, short-lived container with no network,
capped memory/CPU, a read-only root filesystem, and a hard wall-clock
timeout — then the container is removed. The runner is injectable so tests
can exercise the policy without a Docker daemon.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

LANGUAGE_COMMANDS: dict[str, list[str]] = {
    "python": ["python3", "-c"],
    "bash": ["bash", "-c"],
}


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class SandboxExecution:
    """Tool-style class executing untrusted Python/Bash in isolation."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: float = 30.0,
        memory_limit: str = "256m",
        cpu_limit: float = 1.0,
        network: bool = False,
        runner: Callable[..., asyncio.Future] | None = None,
    ):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network = network
        self._runner = runner  # test seam: async fn(cmd: list[str]) -> SandboxResult

    def build_command(self, code: str, language: str = "python") -> list[str]:
        if language not in LANGUAGE_COMMANDS:
            raise ValueError(f"Unsupported language: {language!r}")
        cmd = [
            "docker",
            "run",
            "--rm",
            "--memory",
            self.memory_limit,
            "--cpus",
            str(self.cpu_limit),
            "--read-only",
            "--tmpfs",
            "/tmp:size=64m",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "nobody",
        ]
        if not self.network:
            cmd.extend(["--network", "none"])
        cmd.append(self.image)
        cmd.extend(LANGUAGE_COMMANDS[language])
        cmd.append(code)
        return cmd

    async def _run_subprocess(self, cmd: list[str]) -> SandboxResult:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxResult(
                exit_code=-1, stdout="", stderr="execution timed out", timed_out=True
            )
        return SandboxResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    async def execute(self, code: str, language: str = "python") -> SandboxResult:
        """Run ``code`` in a fresh isolated container and return the result."""
        cmd = self.build_command(code, language)
        runner = self._runner or self._run_subprocess
        return await runner(cmd)
