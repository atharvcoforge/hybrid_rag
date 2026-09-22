FROM python:3.11-slim-bookworm

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf TOKENIZERS_PARALLELISM=false \
    INDEX_DIR=/index CORPUS=/app/documents \
    GOLDEN=/app/evals/policy.jsonl EVAL_OUT=/app/evals/latest.json \
    GENERATOR_URL=http://host.docker.internal:8081/v1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src ./src
COPY documents ./documents
COPY evals ./evals
RUN pip install --no-cache-dir .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
