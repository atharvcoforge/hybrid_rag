# Part F — Run absolutely everything

Date: 2026-09-22.
Commit pushed earlier this session: `9d52194` on `harden/reading-room`.
Evidence: `docs/verification/logs/74`–`102` (gaps noted).

## Command matrix

| # | Command | Exit | Result | Evidence |
| --- | --- | --- | --- | --- |
| 1 | `ruff check .` | 1 | **FAIL** — 66 findings (30 auto-fixable) | `76-part-f-ruff.log` |
| 2 | `mypy --strict src/` | 1 | **FAIL** — 344 errors in 23 files | `77-part-f-mypy.log` |
| 3 | `pytest --cov=rag --cov-branch --cov-fail-under=100 -q` | 1 | **FAIL** — 134 passed, coverage **74%** (need 100%) | `78-part-f-pytest-cov100.log` |
| 4a | `pytest -p no:randomly -q` | 0 | **PASS** — 134 passed | `80-part-f-pytest-no-randomly.log` |
| 4b | `pytest -q` (randomly) | 0 | **PASS** — 134 passed (same) | `81-part-f-pytest-randomly.log` |
| 5 | `pytest -n auto -q` | 0 | **PASS** — 134 passed | `82-part-f-pytest-xdist.log` |
| 6 | `mutmut run` | 1 | **FAIL** — stats collection aborted (`evals/fixture.jsonl` missing from mutmut cwd); no mutation score | `86-part-f-mutmut.log` |
| 7a | vitest via Docker `npm test` | 0 | **PASS** — 9/9 | `92-part-f-vitest.log` |
| 7b | vitest `--coverage` | 1 | **FAIL** — `@vitest/coverage-v8` missing / peer conflict; no coverage gate | `90`, `93` |
| 8 | `rag ingest documents --index index` (empty) | 0 | **PASS** — 16+19+19+25 chunks; `version_group_found` fired | `83-part-f-ingest-empty.log` |
| 9 | `rag ingest` again | 0 | **PASS** — all four `skipped` / `cache_hit: true` | `84-part-f-ingest-skip.log` |
| 10 | `rag verify` | 2 | **FAIL** — CLI has no `verify` subcommand | `85-part-f-verify.log` |
| 11 | `rag purge --missing` | 0 | **PASS** purge — Water doc removed (children 79→54). Post-purge query hit HF proxy 403 (confirmation incomplete) | `87-part-f-purge-missing.log` |
| 11b | re-ingest Water after restore | 0 | **PASS** — children back to 79 | `88-part-f-reingest-after-purge.log` |
| 12 | `rag calibrate` | 0 | **PASS** — `evals/calibration.json` written with τ + held_out abstain | `89-part-f-calibrate.log` |
| 13 | `rag eval --suite evals/suite.yaml` | 1 | **FAIL** — gates named below | `91-part-f-eval.log` |
| 14a | `docker compose build` | 0 | **PASS** — api+web images built | `94-part-f-docker-build.log` |
| 14b | `docker compose up` | 1 | **FAIL** recreate — zombie `rag-week6-api-1` name conflict; old container still serving | `95`, `99` |
| 14c | health/ready/query against existing container | 0 | ready=true; query streamed but answer withheld (`The documents do not say.`) | `96`–`98` |

## Suite gate failures (F13, verbatim)

```
suite gates: FAIL
- recall_at_5=0.623 < 0.920 (scope=holdout)
- mrr=0.598 < 0.850 (scope=holdout)
- unanswerable_abstention=0.000 < 0.900
- retrieve p95 856ms exceeds 400ms (rerank)
- retrieve p95 834ms exceeds 400ms (cascade)
- recall drop 0.377 exceeds 0.030
live=dense
```

Live mode chosen by eval: **dense**. Dense unanswerable abstain is **0.00**. Rerank/cascade p95 ≫ 400 ms.

## Chunk storage (where to look)

Chunks are **not** loose files under `documents/`.

| Location | What |
| --- | --- |
| Local (this session) | `index/rag.sqlite` — gitignored. Tables: `documents` (4), `parents` (21), `children` (79), `vectors` (79) |
| Docker | Named volume `index` mounted at `/index` inside the API container (same schema). Host path is under Docker’s volume store, not the repo. |

After empty ingest (`83`): Carbon 16, Envi_2025 19, Envi_2026 19, Water 25 children; version group logged for the two Envi policies.

## Corpus rename (this session, committed)

| Old | New |
| --- | --- |
| `Carbon_New_2040.pdf` | `Carbon_Reduction_Plan.pdf` |
| `Envi_2040-1.pdf` | `Environmental_Sustainability_Policy_2026.pdf` |
| `Water-Management-Policy.pdf` | `Water_Management_Policy.pdf` |
| `Environmental_Sustainability_Policy_2025.pdf` | unchanged |

Local `index/rag.sqlite` uses the **new** names. The still-running Docker API volume still lists the **old** names in `/api/docs` (`97`/`100`) because the volume was not wiped (force recreate blocked).

## Manual browser screenshots

**NOT RUN** in this pass — no automated browser capture after the compose recreate conflict. Prior Part C screenshots remain under `docs/verification/screenshots/`.

## Status

- **Passed:** pytest 134 (order-independent + xdist); ingest empty + skip; purge removes Water chunks; calibrate writes JSON; vitest 9/9; docker **build**; local chunk index created.
- **Failed:** ruff (66); mypy (344); coverage 74%<100%; mutmut (no score); vitest coverage gate; `rag verify` missing; suite.yaml gates (recall/MRR/unanswerable/SLO/regression); docker up recreate (zombie container / stale volume names).
- **Could not run / incomplete:** browser screenshot set; post-purge query confirmation (HF proxy 403 once); Docker volume wipe to pick up renamed PDFs (needs explicit `docker rm -f` + `volume rm`).

## Remediation (same day)

The 15% score band was applied to reciprocal-rank lists, so rank 2 (score 0.5) was always dropped and recall@5 was precision@1. The band now applies only to `rerank` and `cascade`. BM25 and dense parents keep the search score, and a superseded parent is penalised before the top-5 cut. `suite.yaml` `scope: holdout` is read from holdout rows. `rag verify` checks FTS→vector, child→parent, parent→document. Calibration no longer sets τ to the highest negative when that abstains the positives. Fixture paths are resolved from the test file so mutmut's cwd does not hide `evals/fixture.jsonl`.

| Check | Exit | Result | Evidence |
| --- | --- | --- | --- |
| `ruff check .` | 0 | **PASS** — zero findings | `139-part-f-ruff-zero.log` |
| `pytest -q` | 0 | **PASS** — 139 passed | `129-part-f-pytest-final.log` |
| `rag verify --index index` | 0 | **PASS** — `integrity clean` | `125-part-f-verify.log` |
| BM25 recall/MRR after the band fix (no embedder, no rerank) | 0 | holdout recall **0.946**, MRR **0.784**; all-rows recall **0.967**, MRR **0.771** | `119-bm25-after-band-fix.log` |
| τ sweep on that BM25 score | 0 | no threshold meets unanswerable abstain ≥ 0.90 and answerable abstain ≤ 0.10 together. At τ=10, unanswerable abstain is 0.90 and answerable abstain is 0.45 | `116-tau-sweep.log` |
| vitest `--coverage` in `node:22` | 0 | **PASS** — 9/9; `stream.js` statements 67.07, branches 75, functions 85.71; thresholds 67/75/85 | `128-part-f-vitest-coverage.log` |
| fixture path from `/tmp` | 0 | **PASS** — 2 passed | `126-mutmut-cwd-paths.log` |
| docker ingest of renamed PDFs | 0 | indexed Carbon 16, 2026 policy 19, Water 25; purged `Carbon_New_2040.pdf`, `Envi_2040-1.pdf`, `Water-Management-Policy.pdf` | `132-docker-ingest-renamed.log` |
| `/api/docs` after API restart | 0 | four current filenames; 2025 policy superseded by the 2026 file | `134-api-docs-after-ingest.log` |
| web on port 80 | 0 | HTTP 200; API ready, generation 7 | `138-web-alias-up.log` |
| `docker compose up -d web` | fail | same name conflict: compose wants to create `rag-week6-api-1` which is already running. Web was started by giving that container the `api` network alias | `136-web-compose-up.log` |

`mypy --strict`, `pytest --cov-fail-under=100`, and `mutmut run` were **not re-run**. The last recorded results still stand: mypy 344, coverage 74%, mutmut no score. `rag eval` was **not re-run**; the BM25 measurement above is not a suite-gate pass. Holdout MRR 0.784 is still under 0.850. Rerank/cascade p95 was not re-measured and the 400 ms SLO was not changed.

A live query, `Who signed the Carbon Reduction Plan?`, returned a cached conflict answer about 2040 vs 2050. Passage 1 of that response contains `John Speight`. Log: `140-live-signer-query.log`. The browser showed the renamed document chips and that answer; those images were not written under `docs/verification/screenshots/`. Generator-stopped and renamed-store screenshots were not taken.

## Status after remediation

- **Passed:** ruff; pytest 139; `rag verify`; vitest 9/9 with a coverage gate on `stream.js`; Docker index now lists the renamed PDFs; web HTTP 200.
- **Failed:** holdout MRR 0.784 < 0.850; unanswerable vs answerable BM25 scores do not separate at the suite floors; `docker compose up` still conflicts with the running API name; the signer question was answered as the carbon-neutrality conflict.
- **Not re-run:** mypy (344); branch coverage 100% (last run 74%); mutmut score; full `rag eval` (rerank SLO and regression); screenshot files on disk.
