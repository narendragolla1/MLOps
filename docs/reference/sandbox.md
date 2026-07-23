# omniai.sandbox

## `SandboxExecution`

```python
SandboxExecution(
    image="python:3.11-slim",
    timeout=30.0,             # wall-clock seconds; process killed on expiry
    memory_limit="256m",
    cpu_limit=1.0,
    network=False,            # True re-enables networking (avoid)
    runner=None,              # test seam: async (cmd: list[str]) -> SandboxResult
)
```

- `await execute(code: str, language="python") -> SandboxResult` — runs code in a fresh, locked-down, auto-removed Docker container. Languages: `"python"`, `"bash"` (`ValueError` otherwise).
- `build_command(code, language) -> list[str]` — inspect the exact `docker run` invocation (`--rm --network none --memory --cpus --read-only --tmpfs /tmp --security-opt no-new-privileges --user nobody`).

## `SandboxResult` (dataclass)

`exit_code`, `stdout`, `stderr`, `timed_out`; `ok` property = exit 0 and not timed out.
