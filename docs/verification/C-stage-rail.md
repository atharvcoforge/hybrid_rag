# Part C — Stage rail UI

Date: 2026-09-22.
Evidence: `docs/verification/logs/24`–`30`, screenshots `docs/verification/screenshots/C-*.png`.

## What was built

- Backend emits live SSE `stage` events (via stage bus + worker thread in `/api/query`).
- UI stage rail under the ask box: pending / running / done / slow / failed (+ skipped labels).
- Running stages count up with tabular nums; soft-budget → amber (`slow`); failed stages open an error panel with `trace_id`.
- Copyable `trace_id`; developer **Spans** drawer (off by default) with span table.
- Live announcements in `role="status"` (`.sr-status`); answer text no longer `aria-live`.
- `prefers-reduced-motion`: no caret pulse / no stage animations.
- Honest skips: `skipped — not in mode` (rerank on rrf), `skipped — unavailable` when extractive / generator down.

## Verification

| Check | Result | Evidence |
| --- | --- | --- |
| Python unit tests | PASS 21 | `24-part-c-py-tests.log` |
| Web vitest | PASS 9 | `25-part-c-web-tests.log` |
| SSE `stage` events | PASS | `27-part-c-sse-stages.log` |
| Screenshot normal query (+ spans drawer) | PASS | `C-normal-query.png` |
| Screenshot generator down | PASS | `C-generator-down.png` — banner + `first token/generate skipped — unavailable` |
| Screenshot slow stage | PASS | `C-stage-slow.png` — `first token` 2624ms (>1200 soft) |

## Status

- **Passed:** live stage rail from SSE; trace copy; spans drawer; a11y status region; generator-down honesty; vitest 9/9.
- **Failed:** none for Part C acceptance checks.
- **Notes:** Mid-flight amber “slow” is brief; the saved slow screenshot shows completed timings with TTFT still over soft budget. Llama was stopped briefly for the generator-down shot and restored afterward.
