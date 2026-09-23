# Bring-up — commands that worked (Part A)

Recorded on **2026-09-22** against commit `ab82d31458879b62fb1fee91a1e5988323819374` (`harden/reading-room`).
Evidence logs: `docs/verification/logs/01-*.log` … `13-*.log`.

## Environment

| Item | Value |
| --- | --- |
| OS | Darwin 25.6.0 (macOS 26.6.2), arm64 |
| RAM | 24 GB (`hw.memsize`) |
| Disk free | ~715 GB on `/` |
| System Python | 3.9.6 (`/usr/bin/python3`) — **too old** |
| Project Python | 3.11.16 via `uv` (`uv venv --python 3.11`) |
| CUDA | not used |
| MPS / torch (host clean venv after install) | torch 2.14.0 installed; MPS not probed in clean venv log |
| Docker | 29.8.0 / Compose v5.4.0 |
| Node/npm on host | **MISSING** — UI verified via Compose service on `:80` |
| llama.cpp | already listening on `127.0.0.1:8081` (`{"status":"ok"}`) with `models/qwen2.5-3b-instruct-q4_k_m.gguf` |

## 1. Clean checkout → fast tests

```bash
TMPDIR=$(mktemp -d /tmp/rag-week6-bringup-XXXXXX)
git clone --branch harden/reading-room /path/to/RAG-WEEK6 "$TMPDIR/RAG-WEEK6"
cd "$TMPDIR/RAG-WEEK6"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
```

**Observed** (`docs/verification/logs/02b-clean-checkout-tests.log`):

- `uv_pip_exit=0` (62 packages, including torch)
- `118 passed in 3.88s` — `pytest_exit=0`, wall ~5s for pytest

Note: bare `pip` was not on PATH after `uv venv`; use `uv pip` or `python -m pip`.

## 2. Stack up (documented path)

README / Makefile:

```bash
# generator (once)
make model          # or ensure models/qwen2.5-3b-instruct-q4_k_m.gguf exists
# llama-server on :8081 (Makefile `up` starts it if down)
curl -sf http://127.0.0.1:8081/health

docker compose up --build -d
```

**Observed** (`06-rebuild-stack.log`): `compose_up_exit=0`, `compose_up_seconds=245`. API became `healthy`; web started.

### Health / ready (after rebuild of current sources)

```bash
curl -sS http://127.0.0.1:8000/api/health
# HTTP 200 — example body:
# {"dense_ok":true,"fts_ok":true,"rerank_ok":true,"nli_ok":true,"writer_ok":true,
#  "warmed":true,"index_generation":0,...,"index":true,"writer":true,"circuit":"closed",...}

curl -sS http://127.0.0.1:8000/api/ready
# HTTP 200 — {"ready":true,"index_generation":...}
# HTTP 503 while warming — {"detail":"warming up"}
```

### Query body field

The request model field is **`q`**, not `question`:

```bash
curl -sS -N -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"q":"By when does Coforge reach carbon neutrality in operations?"}'
```

## 3. Entrypoint ingest

`docker/entrypoint.sh` no longer gates on `side.sqlite`. It runs `python -m rag ingest` on every boot. A second boot skips unchanged files (`ingest_skip` / `cache_hit: true`). A file added under the corpus is indexed on the next ingest. Local proof: `docs/verification/logs/152-entrypoint-incremental.log` (`one.md` skipped, `throwaway.md` indexed).

## 3b. Historical defect: entrypoint gated on `side.sqlite`

Recorded against the Part A commit, before the entrypoint change in section 3. The script at that commit was:

```sh
if [ ! -f "$INDEX_DIR/side.sqlite" ]; then
  python -m rag ingest "$CORPUS" --index "$INDEX_DIR"
fi
```

**Observed** (`09-empty-index-diagnosis.log`, `12-index-reuse-and-throwaway.log`):

- Volume already contained legacy `/index/side.sqlite` → entrypoint **skipped** ingest.
- Fresh `/index/rag.sqlite` was empty → first queries returned `hits: []` and empty `answer`.
- After manual ingest + API restart, index had 4 documents / 80 children and queries returned hits + SSE tokens.
- Second boot: `rag.sqlite` mtime unchanged, `ingest_lines=0` → reuses existing index (no re-ingest).
- Throwaway `documents/ZZXQ_bringup_throwaway.txt` added and API restarted: **`throwaway_present False`** — new file **not** indexed. The Part A acceptance that “adding a new file does get indexed on next boot” **FAILED**. Defect is **not** gone.

### Workaround that produced a real streamed answer

```bash
docker exec rag-week6-api-1 python -m rag ingest /app/documents --index /index
docker compose restart api
# wait for /api/ready == 200
curl -sS -N -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"q":"By when does Coforge reach carbon neutrality in operations?"}'
```

**Observed** (`10-manual-ingest-and-query.log`, `11-restart-after-ingest-query.log`):

- Ingest: `indexed` 4 PDFs (16+19+19+26 parent counts logged), `ingest_seconds=126`
- After restart: SSE `event: meta` with citations (including both Envi 2040 and 2025 policy), then tokens; final gated answer text was `The documents do not say.` (`verification.state=withheld`, `reason=missing_citation`)

## 4. Web UI

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1/
# 200
```

UI loads “Reading room” (`docs/verification/screenshots/A-ui-load.png`). Host has no Node/npm; production UI served by Compose `web` (nginx).

Browser CDP console capture: **NOT RUN** — CDP `Runtime.evaluate` failed with a harness error in this session. No JS exception was visible from the accessibility snapshot (page rendered).

## 5. Clean clone → first answered query (timing)

Single stopwatch from empty clone through first non-empty answer was **not** completed as one continuous command (volume wipe `docker compose down -v` was blocked by the environment approval gate).

Measured segments that did run:

| Segment | Seconds | Log |
| --- | ---: | --- |
| Clean clone + `uv pip install -e ".[dev]"` + `pytest -q` | ~13 (pytest 5; install dominated earlier download) | `02b-…` |
| `docker compose up --build -d` (rebuild) | 245 | `06-…` |
| Manual `rag ingest` in container | 126 | `10-…` |
| API restart + ready + streamed query | ~10 | `11-…` |

Lower bound if HF/torch layers cached and llama already up: **≈ 245 + 126 + 10 ≈ 6.4 minutes** after images exist — under the dossier’s ten-minute bar **only if** ingest actually runs. With a leftover `side.sqlite`, documented `docker compose up` alone does **not** yield an answered query without the manual ingest workaround.

## 6. Follow-ups from Part A

1. The `side.sqlite` gate is gone. Boot calls incremental ingest. See section 3 and `152-entrypoint-incremental.log`.
2. Document `q` in README examples. Done in the README query example.
3. Install Node on the host if local `npm run dev` is required; Compose path works without it.
