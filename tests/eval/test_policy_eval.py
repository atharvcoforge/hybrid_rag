"""Run the policy golden set through the real embedder. Not part of the unit job."""

import os
from pathlib import Path

import pytest

from rag.embed import encode_query, rerank_scores
from rag.evaluate import (
    answer_hit_rate,
    check_gates,
    evaluate,
    load_rows,
    load_split,
    load_suite,
    pick_live,
    row_hit,
)
from rag.generate import complete, writer_up
from rag.models import EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION
from rag.pipeline import ingest
from rag.retrieve import retrieve
from rag.store import Index

pytestmark = pytest.mark.eval

ROOT = Path(__file__).resolve().parents[2]


def test_policy_set_clears_the_suite_and_hybrid_beats_dense(tmp_path: Path) -> None:
    suite = load_suite(ROOT / "evals" / "suite.yaml")
    rows = load_rows(suite["rows"])
    split_path = ROOT / (suite.get("split") or "evals/split.json")
    split = load_split(split_path) if split_path.exists() else None
    index_dir = tmp_path / "index"
    list(ingest(ROOT / "documents", index_dir))

    index = Index(index_dir, EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION)
    index.open()
    try:
        def ask(mode: str, row: dict) -> object:
            return retrieve(index, row["q"], encode_query, rerank=rerank_scores, mode=mode)

        lines, _fitted = evaluate(index, rows, ask, split=split)
        live = pick_live(lines, p95_limit=(suite.get("slo") or {}).get("retrieve_p95_ms"))
        named = _hybrid_beats_dense(rows, ask)
        answers = _answers(rows, ask, live)
    finally:
        index.close()

    by_mode = {(line.mode, line.kind): line for line in lines}
    dense = by_mode[("dense", "holdout")].recall
    hybrid = by_mode[("rrf", "holdout")].recall
    assert hybrid > dense, f"hybrid recall {hybrid:.3f} did not beat dense {dense:.3f}"
    assert named, "no holdout question where dense missed the top 5 and hybrid hit"
    print(f"hybrid beats dense on {named['id']}: {named['q']}")

    if not writer_up():
        pytest.fail("generator is down; answer checks were not run")
    rate = answer_hit_rate(answers[1], answers[0])
    assert rate is not None and rate >= 0.90
    failures = check_gates(
        suite,
        lines,
        rows=rows,
        live_mode=live,
        split=split,
        answers=answers[0],
    )
    assert not failures, failures


def _hybrid_beats_dense(rows: list[dict], ask) -> dict | None:
    for row in rows:
        if row.get("kind") == "unanswerable" or not row.get("must_contain"):
            continue
        dense = ask("dense", row)
        hybrid = ask("rrf", row)
        if not row_hit(dense.hits, row) and row_hit(hybrid.hits, row):
            return row
    return None


def _answers(rows: list[dict], ask, live: str) -> tuple[dict[str, str], list[dict]]:
    from rag.cli import _gated_text
    from rag.gates import check_all

    chosen = rows
    if os.environ.get("RAG_ANSWER_ROWS") == "ci":
        chosen = [row for row in rows if row.get("ci")]
        if len(chosen) < 8:
            raise AssertionError(f"ci answer subset has {len(chosen)} rows")
    answers: dict[str, str] = {}
    for row in chosen:
        result = ask(live, row)
        text, _gate = _gated_text(row["q"], result, complete, check_all)
        answers[row["id"]] = text
    return answers, chosen
