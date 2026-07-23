# OmniAI gateway image (the serving engine runs in its own container).
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY omniai ./omniai
RUN pip install --no-cache-dir --prefix=/install ".[postgres]"

FROM python:3.11-slim

RUN useradd --create-home --uid 10001 omniai
COPY --from=builder /install /usr/local

USER omniai
WORKDIR /home/omniai
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/health/live')" || exit 1

CMD ["uvicorn", "omniai.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
