# Part E — Eval suite and YAML config

Date: 2026-09-22.
Evidence: `docs/verification/logs/65`–`73`.

## Deliverables

| Item | Result | Evidence |
| --- | --- | --- |
| `evals/suite.yaml` with corpus sha256 pins, splits, gates, SLO, regression | PASS | `65-corpus-sha256.log`, `71-part-e-suite-gates-keys.log` |
| Eight `conflict` rows in `evals/policy.jsonl` (one per Part D fact) | PASS (8) | `66-append-conflict-injection-rows.log`, `73-part-e-conflict-rows.log` |
| `injection` kind with real rows | PASS (3) | `73-part-e-conflict-rows.log` |
| Kind coverage: lexical, semantic, multi-hop, unanswerable≥25, adversarial, conflict, injection | PASS | `69-part-e-kind-coverage.log` |
| `multi-hop` non-empty | PASS (15 rows, already present) | `69-part-e-kind-coverage.log` |
| Rename `evals/golden.jsonl` → `evals/fixture.jsonl` | PASS (already done; no code refs to `golden.jsonl`) | `69`, `73` |
| Stratified split regenerated from suite seed `20260922` | PASS — calibration 60 / holdout 92 | `67-regenerate-split.log` |
| Evaluator + CI read suite.yaml | PASS | `src/rag/evaluate.py`, `src/rag/cli.py`, `.github/workflows/ci.yml` |
| Unit tests | PASS 11 (`test_evaluate`) / 20 related | `70`, `72` |

## Kind counts (post-change)

```
lexical: 48
semantic: 33
multi-hop: 15
unanswerable: 30
adversarial: 15
conflict: 8
injection: 3
total: 152
```

## Conflict rows (schema)

Each row carries `current_value`, `superseded_value`, `superseded_doc`, `disclose: true`. A row passes only via `conflict_pass`: answer must contain the current value **and** name the conflict (superseded value + disclosure marker or superseded filename).

| id | current | superseded |
| --- | --- | --- |
| conflict-001 | 10th March 2026 | 10th March 2025 |
| conflict-002 | 1st April 2025 | 1st April 2024 |
| conflict-003 | 2040 | 2050 |
| conflict-004 | 2025 | 2027 |
| conflict-005 | 2030 | 2040 |
| conflict-006 | 2025 | 2030 |
| conflict-007 | 2040 | 2045 |
| conflict-008 | 2026 | 2025 |

## suite.yaml corpus pins (verified)

| path | sha256 (prefix) |
| --- | --- |
| Carbon_New_2040.pdf | d296911be1a5… |
| Envi_2040-1.pdf | be80cf8f28a3… (current, review_date 2026-03-10) |
| Environmental_Sustainability_Policy_2025.pdf | 92061826311c… (superseded) |
| Water-Management-Policy.pdf | 8c0b22d1ac47… |

`verify_corpus(suite)` returned `[]` (empty = all pins match).

## Code wiring

- `load_suite`, `verify_corpus`, `make_split` / `make_split_from_suite`, `conflict_pass`, `injection_pass`, `check_gates`, `assert_kind_coverage` in `src/rag/evaluate.py`
- `rag eval --suite evals/suite.yaml` (default): verifies pins + kinds, scores retrieval, reports `suite gates: PASS|FAIL`
- Answer gates `conflict_disclosure` / `injection_resisted` require generated answers (`answers=` map); retrieval-only eval skips them — filled in Part F
- CI step “Suite corpus pins and kinds” before pytest
- `pyyaml>=6.0` added as a direct dependency

## Status

- **Passed:** suite.yaml; 8 conflict + 3 injection rows; all required kinds with coverage; fixture rename confirmed; split regenerated; unit tests; corpus pins.
- **Failed:** none for Part E structural checks.
- **Notes:** Full `rag eval` gate pass (recall/MRR/groundedness/conflict live answers) is Part F. Holdout unanswerable count is now 18 under the suite’s 0.4/0.6 stratified split (total unanswerable remains 30 ≥ 25). YAML needed a space after `unanswerable_abstention:` or PyYAML mis-parsed the key (`71` after fix).
