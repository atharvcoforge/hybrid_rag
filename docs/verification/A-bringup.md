# Part A — Prove it runs from nothing

Date: 2026-09-22. Commit: `ab82d31458879b62fb1fee91a1e5988323819374`.
Deliverable: `docs/BRINGUP.md`. Logs: `docs/verification/logs/01`–`13`.

## Checklist

| Step | Result | Evidence |
| --- | --- | --- |
| Environment recorded | PASS | `01-environment.log` — Darwin arm64, 24 GB RAM, ~715 GB free, Python 3.9 system / 3.11 via uv, Docker ok, node/npm missing on host |
| Clean checkout + fresh venv + fast tests | PASS | `02b-clean-checkout-tests.log` — `118 passed in 3.88s` |
| `/api/health` 200 expected body | PASS (after rebuild) | `07-post-rebuild-ready.log` |
| `/api/ready` models warmed + index | PASS (after rebuild) | `07-…` — `{"ready":true,"index_generation":…}`; pre-rebuild running image returned **404** (`04-…`) |
| Web UI loads | PASS | `04-…` HTTP 200; screenshot `docs/verification/screenshots/A-ui-load.png` |
| UI console errors | NOT RUN | CDP evaluate failed in harness; no crash visible in snapshot |
| One real streamed cited query | PASS (after manual ingest + restart) | `11-restart-after-ingest-query.log` — SSE meta hits + token + done (gated answer) |
| Clean clone → first answer < 10 min | PARTIAL | Segments sum ≈ 6.4 min with cached layers; continuous stopwatch NOT RUN (`docker compose down -v` blocked). Documented compose-only path fails when `side.sqlite` present |
| Second boot reuses index | PASS | `12-…` — mtime unchanged, `ingest_lines=0`, still 4 docs / 80 children |
| New file indexed on next boot | **FAIL** | `12-…` — `throwaway_present False`; entrypoint `side.sqlite` gate still present |

## Status

- **Passed:** env capture; clean-venv pytest (118); compose rebuild; health/ready after rebuild; UI HTTP load; index reuse on restart; streamed query after manual ingest.
- **Failed:** entrypoint still skips ingest when `side.sqlite` exists; throwaway corpus file not indexed on reboot; pre-rebuild `/api/ready` 404 (stale image).
- **Could not run:** host `npm` UI path (no Node); full `docker compose down -v` clean-volume timing (approval blocked); browser console CDP dump.
