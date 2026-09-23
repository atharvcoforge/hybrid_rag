# Part B — Stage-level logging

Date: 2026-09-22. Commit under test: workspace with Part B changes (uncommitted).
Evidence: `docs/verification/logs/14`–`23`.

## What was built

- `src/rag/telemetry.py` — `trace_id` (ContextVar + thread-local for SSE generators), UUIDv7 minting, inbound `X-Request-ID` / `X-Trace-ID`, JSON or `--log-format=console` / `LOG_FORMAT`, per-module `LOG_LEVEL_*`, `LOG_CONTENT=1` for query/passage fields, `span()` with `start` / `finish` / `error` / `stage_slow`, soft+hard budgets, `StageBudgetExceeded`.
- Wired through retrieve (`embed_query`…`gate`), generate (`generate_first_token` / `generate_complete`), post-gen gates (`gate_form`…`gate_coverage`), ingest (`parse`…`embed` + file/page progress).
- API: first SSE event `trace`, response headers `X-Request-ID` / `X-Trace-ID`.

## Verification checklist

| Check | Result | Evidence |
| --- | --- | --- |
| Unit tests (telemetry + server) | PASS 21 | `14-part-b-unit-tests.log` |
| Full pytest | PASS 127 | `15-…`, `22-part-b-final-pytest.log` |
| One query complete span trace | PASS | `21-part-b-live-traces.log`, `23-part-b-success-timeline.jsonl` — stages in order with `trace_id=partb-success-trace-002` |
| Dead generator stage obvious | PASS | `20-…` local + `21-…` live `URLError` on `generate_first_token` / `generate_complete` |
| `stage_slow` during hang | PASS | Unit `test_stage_slow_fires_during_hang`; live `generate_first_token` `stage_slow` at ~1200ms soft budget while still running (`21-…`) |
| No passage/query text at default level | PASS | `20-part-b-local-verify.log`; live `query_start` has no `query` field |

### Success timeline (excerpt)

```
embed_query start → finish
dense start → finish (candidates_out=20)
lexical start → finish
fuse start → finish (top_score=…)
gate start → finish
generate_complete start
generate_first_token start → stage_slow → finish
generate_complete finish
gate_form start → finish   # failed form → later gates not entered
```

### Dead generator (excerpt)

```
generate_complete start
generate_first_token start → error URLError Connection refused
generate_complete error URLError
```

## Status

- **Passed:** structured spans with lifecycle; trace id on SSE + headers; soft-budget `stage_slow` while running; hard-budget unit test; dead-gen failure obvious from logs alone; content scrub; 127 pytest green.
- **Failed:** none observed for Part B acceptance checks.
- **Could not run / notes:** cascade `rerank` span not in the live rrf trace (mode has no rerank arm); `gate_literal` / `gate_entail` / `gate_coverage` only run when `gate_form` passes — withheld answers stop at form (visible in the success trace).
