# Part D — Planted data-quality / conflict handling

Date: 2026-09-22.
Evidence: `docs/verification/logs/31`–`64`, plus unit logs `49`–`52`, `63`.

## PDF ground truth (verified before tests)

All eight conflict rows match the PDFs (`31-pdf-conflict-facts.log`). RE100 “100% renewable by 2050” is present in both and is **not** a conflict.

| Fact | Stale `Environmental_Sustainability_Policy_2025.pdf` | Current `Envi_2040-1.pdf` |
| --- | --- | --- |
| Review date | 10th March 2025 | 10th March 2026 |
| Last review | 1st April 2024 | 1st April 2025 |
| Carbon Neutral in operations by | 2050 | 2040 |
| 10% green electricity by | 2027 | 2025 |
| ~50% green electricity by | 2040 | 2030 |
| EVs 10% by | 2030 | 2025 |
| EVs 50% by | 2045 | 2040 |
| Copyright | © 2025 | © 2026 |

## Three defects — proven before the fix

### 1. Hardcoded doc lists vs live index

Evidence: `32-defect1-hardcoded-lists.log`, `36-defect1-and-rrf-dupes.log`.

- Corpus / index: **4** PDFs including both Envi versions.
- `DOC_FILTERS` in `App.jsx`: **3** pills — included the stale 2025 file as “Environmental Sustainability Policy”, **omitted** `Envi_2040-1.pdf`.
- `TITLES` in `server.py`: hardcoded map (4 entries after local drift).
- `/api/docs` did not exist.

### 2. Silent pick / silent abstain on carbon neutrality

Evidence: `33-defect2-silent-pick.log`, `37-defect2-silent-pick-with-generator.log`, `40-defect2-full-answer.log`.

- Retrieval ranked **stale 2050** and **current 2040** as co-equal top hits (`score` identical).
- With generator up: model answered **"The documents do not say."** with both conflicting passages in `hits` and **no disclosure**.
- Target behaviour (2040 + name superseded 2050) was **not** met.

### 3. Near-identical chunks compete in RRF

Evidence: `36-defect1-and-rrf-dupes.log`.

- Top-20 child RRF list: **3** exact-masked cross-`doc_id` duplicate groups (6/20 rows) spanning the two Envi PDFs; ranks 1–2 are near-identical conflicting carbon/energy chunks.

## What was built

- **Ingest** (`rag.versions.reconcile_versions`): chunk-overlap version groups, `review_date`, `status` current/superseded, `supersedes` / `superseded_by`; loud `version_group_found` log.
- **Retrieval**: superseded hits down-ranked but retained; score-band keep also re-attaches one superseded sibling per group so conflict stays visible.
- **Gate** `gate_conflict` after coverage: answer from current (2040), disclose superseded 2050.
- **API** `GET /api/docs` from index; hit payloads carry `superseded` / `version_group` / titles from DB.
- **UI**: filters from `/api/docs`; superseded passage chip; conflict banner above the answer.

## Post-fix verification

| Check | Result | Evidence |
| --- | --- | --- |
| PDF fact table | PASS 8/8 | `31-pdf-conflict-facts.log` |
| Version grouping on ingest | PASS — current=`Envi_2040-1.pdf`, superseded=2025 policy, review dates 2026-03-10 / 2025-03-10 | `55-part-d-ingest-version-group.log` |
| `/api/docs` lists 4 docs with status | PASS | `57` / `60-part-d-api-docs*.log` |
| Live target question | PASS | `59-part-d-live-conflict-after-sibling-fix.log` |
| Unit tests (versions/gates/server/retrieve) | PASS 27 | `63-part-d-final-pytest.log` |
| Vitest | PASS 9/9 | `62-part-d-vitest.log` |

### Exact live answer (target question)

> Coforge commits to becoming Carbon Neutral in its operations by 2040 [1]. Note: a superseded version (Environmental_Sustainability_Policy_2025.pdf) states 2050 [4].

Conflict payload: `current_doc=Envi_2040-1.pdf`, `superseded_value=2050`, `current_value=2040`. Hit [4] marked `superseded=true`.

## Status

- **Passed:** PDF verification; three defects evidenced; ingest version grouping; down-rank+retain; conflict gate disclosure; `/api/docs` + UI hooks; unit + vitest.
- **Failed:** none for Part D acceptance checks.
- **Notes:** Live API verification used a hot-patch of `site-packages` matching workspace `src/rag/*` after the image build (sibling-retain fix). Generator required `--host 0.0.0.0` and `GENERATOR_URL=http://192.168.65.254:8081/v1` (hostname→IPv6 was unreachable from the container). Title extraction from chunk text is heuristic (“Sustainability Policy” vs full cover title).
