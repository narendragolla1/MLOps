# Installation

## Core library

```bash
pip install -e .
```

The core installs FastAPI, Pydantic v2, httpx, SQLModel/SQLAlchemy, Alembic, and prometheus-client. It runs with no GPU and no external services (SQLite is the zero-config default database).

## Optional extras

| Extra | Installs | When you need it |
| --- | --- | --- |
| `pip install -e ".[vllm]"` | `vllm` | Serving models with the vLLM backend on your own GPUs. |
| `pip install -e ".[sglang]"` | `sglang` | Serving with the SGLang backend (RadixAttention prefix caching). |
| `pip install -e ".[training]"` | `peft`, `trl`, `transformers`, `datasets`, `torch` | Real LoRA fine-tuning in the [continuous-learning loop](../tutorials/continuous_learning.md). |
| `pip install -e ".[postgres]"` | `asyncpg` | Postgres-backed interaction logging in production. |
| `pip install -e ".[telemetry]"` | `opentelemetry-api`, `opentelemetry-sdk` | Exporting traces via OTLP. |
| `pip install -e ".[dev]"` | pytest, ruff, mypy, coverage | Running the test suite and linters. |

## Environment configuration

All runtime configuration is environment-driven with the `OMNIAI_` prefix (12-factor). The essentials:

```bash
export OMNIAI_API_KEYS=my-secret-key          # required to serve (fail-closed)
export OMNIAI_DATABASE_URL=sqlite+aiosqlite:///interactions.db
export OPENAI_API_KEY=...                     # or ANTHROPIC_API_KEY, for cloud providers
```

The full variable table is in the [Settings reference](../reference/settings.md).

## Verify your install

```bash
pytest          # full suite, no GPU required
```

## Next steps

- Follow the [Quickstart](quickstart.md) to run your first agent.
- Deploying to production? See [Deploy with Docker Compose](../how_to/deploy_docker_compose.md).
