# How to execute untrusted code in a sandbox

LLM-generated code must never run in your API process. `SandboxExecution` runs Python or Bash in a **disposable, locked-down Docker container** per execution.

## Usage

```python
from omniai.sandbox import SandboxExecution

sandbox = SandboxExecution(image="python:3.11-slim", timeout=30.0, memory_limit="256m")
result = await sandbox.execute("print(2 + 2)", language="python")   # or "bash"
result.ok          # exit 0 and not timed out
result.stdout      # "4\n"
result.stderr, result.exit_code, result.timed_out
```

## The lockdown

Every execution launches a fresh container with:

- `--network none` — no network (opt back in with `network=True` only if you must)
- `--memory` / `--cpus` caps
- `--read-only` root filesystem (+ small tmpfs on `/tmp`)
- `--security-opt no-new-privileges`, runs as `nobody`
- `--rm` — the container is destroyed afterward
- a hard wall-clock timeout; on expiry the process is killed and `timed_out=True`

Inspect exactly what runs with `sandbox.build_command(code, language)`.

## As an agent tool

```python
from omniai.graph import tool

@tool
async def run_python(code: str) -> str:
    """Execute Python code and return its output."""
    result = await sandbox.execute(code)
    return result.stdout if result.ok else f"error: {result.stderr or 'timed out'}"
```

## Testing without Docker

Inject `SandboxExecution(runner=...)` — an async callable receiving the built command — to exercise policy and plumbing without a Docker daemon (this is how the framework's own tests run).
