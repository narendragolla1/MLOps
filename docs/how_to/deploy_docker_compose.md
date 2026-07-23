# How to deploy with Docker Compose

The `deploy/` directory ships a three-service production stack:

| Service | Image | Role |
| --- | --- | --- |
| `gateway` | built from `deploy/Dockerfile` (multi-stage, non-root, healthcheck) | This framework: auth, graph, memory, metrics. Applies DB migrations on boot, then serves. |
| `postgres` | `postgres:16-alpine` | Interaction log (persistent volume, healthcheck). |
| `vllm` | `vllm/vllm-openai:latest` | GPU serving with `--enable-lora`; shares an `adapters` volume with the gateway so [LoRA hot-swaps](lora_hot_swap.md) resolve by path. |

## Steps

```bash
cp deploy/.env.example deploy/.env
# edit deploy/.env — at minimum:
#   OMNIAI_API_KEYS=<your keys>
#   POSTGRES_PASSWORD=<strong password>
#   OMNIAI_MODEL=Qwen/Qwen2.5-7B-Instruct   (+ HUGGING_FACE_HUB_TOKEN for gated models)

docker compose -f deploy/docker-compose.yml up -d --build

curl -H "X-API-Key: $KEY" localhost:8080/v1/messages -d '{"content": "hi"}'
```

## What the wiring does

- The gateway runs `omniai.app:create_app` — the production factory that assembles engine (external mode, pointed at `http://vllm:8000`), Postgres-backed buffer, continuous learner, guardrails, and the full security/observability stack from env vars.
- Its entrypoint runs `alembic upgrade head` before uvicorn, so the schema is always current ([migrations guide](database_migrations.md)).
- `depends_on` + healthchecks sequence startup: Postgres and vLLM must be healthy before the gateway starts; `restart: unless-stopped` everywhere.
- vLLM gets a GPU device reservation (`driver: nvidia`) — the host needs the NVIDIA container toolkit. Model downloads cache in the `hf-cache` volume.

## Operations

```bash
docker compose -f deploy/docker-compose.yml logs -f gateway   # JSON logs with request IDs
curl localhost:8080/health/ready                              # readiness (DB + engine)
curl localhost:8080/metrics                                   # Prometheus scrape target
```

Scaling note: the gateway is safe to replicate behind a load balancer, but the in-process rate limiter multiplies limits per replica and background training should move to a dedicated worker at that point — see [auth & rate limiting](auth_rate_limiting.md).
