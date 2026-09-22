# Reading room

Local hybrid RAG over a small showcase corpus of Coforge policy PDFs. Retrieval is dense + BM25 fused in SQLite; answers are generated locally and gated before they ship.

This is still one machine and four documents. Auth, calibration, and the golden set are real — the corpus size is not production scale, and saying so is intentional.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# optional: pip install -e ".[ocr,layout]"

rag ingest documents --index index
rag calibrate   # writes thresholds; eval is read-only after that
rag eval

# API
INDEX_DIR=index CORPUS=documents uvicorn rag.server:app --port 8000

# UI
cd web && npm install && npm run dev
```

Clone to first answered query should be under ten minutes if the embedder weights are already cached in `HF_HOME`.

Docker: `docker compose up --build` (API on `:8000`, web on `:80`). Point `GENERATOR_URL` at a local llama.cpp OpenAI-compatible server.

## What is gated

| Gate | What it catches |
| --- | --- |
| Form | Missing / out-of-range citations, empty or truncated answers |
| Literal grounding | Fabricated figures, dates, units |
| Coverage | Withholds when too little of the answer is supported |

Derived blocks (VLM captions) cannot be the sole support for a factual claim. OCR blocks are flagged in the evidence column.

## Design choices worth knowing

- **One SQLite file** (`index/rag.sqlite`) — exact cosine, weighted FTS5. No Chroma.
- **Parent/child chunking** with optional contextual prefixes behind `--contextual` / `contextual=True`.
- **Table ladder** — ruled lines → borderless text strategy → optional Docling.
- **Reranker protocol** — local cross-encoder by default; `NullReranker` on failure; `JevReranker` stub refuses remote unless `ALLOW_REMOTE_INFERENCE=1`.
- **Resilience** — LRU answer cache keyed by index generation, bounded inference queue (429), generator circuit breaker with extractive fallback.

## Tests and CI

```bash
pytest -q
cd web && npm test
```

CI runs pytest on every push. Mutation score ≥ 0.90 on `chunk` / `retrieve` / `gates` is the release bar; run locally with `mutmut` when changing those modules.

## API

| Route | Notes |
| --- | --- |
| `GET /api/health` | Liveness + degradation messages |
| `GET /api/ready` | 503 until warmup finishes |
| `POST /api/query` | SSE; optional `doc_id` filter |
| `POST /api/ingest` | Bearer token when `API_TOKEN` set; corpus root only |
| `GET/POST /api/eval` | Scores report; POST is mutating |

## Honesty clause

Still not a production service: one machine, a showcase corpus of four policies, and a golden set written by hand. Auth and abstention gates exist so the numbers mean something; they do not turn four PDFs into an enterprise search product.
