FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf TOKENIZERS_PARALLELISM=false \
    UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    INDEX_DIR=/index CORPUS=/app/documents \
    GOLDEN=/app/evals/policy.jsonl EVAL_OUT=/app/evals/latest.json \
    GENERATOR_URL=http://host.docker.internal:8081/v1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY documents ./documents
COPY evals ./evals
RUN uv sync --frozen --no-dev --no-editable

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && useradd --create-home --uid 1000 --shell /bin/sh rag \
    && mkdir -p /index /cache/hf \
    && chown -R rag:rag /app /index /cache/hf

USER rag
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
