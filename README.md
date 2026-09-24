# Reading room

Local hybrid RAG over four Coforge policy PDFs. Dense and BM25 results are fused in SQLite. A local llama.cpp model writes the answer, and gates check it before it is shown.

This is one machine and four documents. The golden set and the abstention gates are real. The corpus is not production scale.

## Quick start

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv sync --frozen --extra dev

rag ingest documents --index index
rag calibrate --index index
rag eval --index index          # full 152-row run; needs the generator on :8081
rag ask "Who signed the Carbon Reduction Plan?" --index index

# retrieval only, one mode
rag query "..." --index index --mode dense
rag query "..." --index index --mode rrf
```

Set `GENERATOR_URL=http://127.0.0.1:8081/v1` when the generator is on the host. The default URL is `host.docker.internal`, for the API container.

```bash
make eval                         # same as rag eval
cd web && npm test && npm run build
docker compose up --build         # API :8000, web :80
```

Measured commands and logs are in `docs/SUBMISSION.md`.

## Design choices

- One SQLite file. Exact cosine and FTS5 BM25. No vector server.
- Parent chunks around 700 characters, child chunks around 180, so a citation can point at the section that contains the fact.
- Reciprocal-rank fusion (`k=60`) is the live retriever. The cross-encoder is measured and is not live: on this corpus its p95 is about 2.2s and a 0–1 threshold abstains most questions. See the decision log.
- Version handling is generic. A superseded file stays in the index, passages are tagged current or superseded, and a conflict note names the older file when the numbers differ.
- The enforced mutation floor is the number in `evals/mutmut_floor.txt`. The latest local run killed 1825 of 2895 decided mutants, ratio 0.630 (`docs/verification/logs/72-mutmut.log`). That measured ratio is the ratchet. It is not 0.90.

## API

| Route | Notes |
| --- | --- |
| `GET /api/health` | Liveness |
| `GET /api/ready` | 503 until warmup finishes |
| `POST /api/query` | SSE. Body field is `q` |
| `POST /api/ingest` | Bearer token when `API_TOKEN` is set |
| `GET/POST /api/eval` | Score report |

## Honesty

Four hand-written policies and a hand-written golden set. Auth and abstention gates keep the numbers meaningful. They do not make this an enterprise search product.
