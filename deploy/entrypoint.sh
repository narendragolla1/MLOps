#!/bin/sh
# Container entrypoint: apply pending DB migrations, then start the gateway.
set -e

echo "applying database migrations..."
alembic upgrade head

exec uvicorn omniai.app:create_app --factory --host 0.0.0.0 --port 8080
