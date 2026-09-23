"""Branch coverage for rag.calibrate. No model downloads."""

import json

from rag.calibrate import calibrate, fit_mode_threshold, write_report
from rag.models import Hit, Retrieval


def _hit(text="yes", score=0.5, doc_id="guide.md"):
    return Hit(
        parent_id="p",
        parent_text=text,
        heading_path="H",
        source_path=doc_id,
        file_sha256="abc",
        page_start=0,
        page_end=0,
        start_char=0,
        end_char=len(text),
        child_id="c",
        score=score,
        confident=True,
    )


class _Index:
    model_id = "m"
    model_revision = "r"

    def __init__(self):
        self.saved = {}

    def set_tau(self, key, value):
        self.saved[key] = value


def test_fit_mode_threshold_keeps_the_best_precision_cut():
    rows = [
        {"kind": "lexical"},
        {"kind": "lexical"},
        {"kind": "lexical", "must_contain": "yes", "doc_id": "guide.md"},
        {"kind": "unanswerable"},
        {"kind": "lexical", "must_contain": "yes", "doc_id": "guide.md"},
        {"kind": "lexical", "must_contain": "yes", "doc_id": "guide.md"},
        {"kind": "lexical", "must_contain": "yes", "doc_id": "guide.md"},
    ]
    results = [
        Retrieval(hits=[]),
        Retrieval(hits=[_hit("ignored", 0.99)]),
        Retrieval(hits=[_hit("nope", 0.05)]),
        Retrieval(hits=[_hit("no", 0.2)]),
        Retrieval(hits=[_hit("yes", 0.9)]),
        Retrieval(hits=[_hit("yes", 0.8)]),
        Retrieval(hits=[_hit("yes", float("nan"))]),
    ]
    assert fit_mode_threshold([], []) is None
    assert fit_mode_threshold(rows, results, keep=0.95) == 0.8


def test_fit_mode_threshold_falls_back_when_nothing_clears_keep():
    rows = [
        {"kind": "lexical", "must_contain": "yes", "doc_id": "guide.md"},
        {"kind": "unanswerable"},
    ]
    results = [
        Retrieval(hits=[_hit("yes", 0.4)]),
        Retrieval(hits=[_hit("no", 0.9)]),
    ]
    assert fit_mode_threshold(rows, results, keep=0.99) == 0.4


def test_fit_mode_threshold_uses_fit_tau_without_negatives():
    rows = [{"kind": "lexical", "must_contain": "yes", "doc_id": "guide.md"}]
    results = [Retrieval(hits=[_hit("yes", 0.7)])]
    assert fit_mode_threshold(rows, results) == 0.7


def test_fit_mode_threshold_returns_none_without_positives():
    rows = [{"kind": "unanswerable"}]
    results = [Retrieval(hits=[_hit("no", 0.4)])]
    assert fit_mode_threshold(rows, results) is None


def test_calibrate_skips_a_mode_that_does_not_fit_and_writes_json(tmp_path):
    rows = [{"id": "t", "kind": "lexical", "must_contain": "yes", "doc_id": "guide.md"}]
    index = _Index()

    def ask(mode, _row):
        if mode == "rrf":
            return Retrieval(hits=[])
        return Retrieval(hits=[_hit("yes", 0.9)])

    report = calibrate(index, rows, ask, modes=("rrf", "rerank"))
    assert report["modes"]["rrf"] == {"tau": None}
    assert report["modes"]["rerank"]["tau"] == 0.9
    assert "held_out_abstain" not in report["modes"]["rerank"]
    assert index.saved
    out = tmp_path / "nested" / "cal.json"
    write_report(out, report)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["modes"]["rerank"]["tau"] == 0.9
    assert loaded["modes"]["rrf"]["tau"] is None


def test_calibrate_reports_held_out_abstention():
    rows = [
        {"id": "t", "kind": "lexical", "must_contain": "yes", "doc_id": "guide.md"},
        {"id": "h1", "kind": "unanswerable"},
        {"id": "h2", "kind": "lexical", "must_contain": "yes", "doc_id": "guide.md"},
        {"id": "h3", "kind": "unanswerable"},
    ]
    index = _Index()

    def ask(mode, row):
        if mode == "rrf":
            return Retrieval(hits=[])
        if row.get("kind") == "unanswerable":
            reason = "no_confident_hit" if row["id"] == "h1" else "answered"
            return Retrieval(hits=[_hit("no", 0.2)], reason=reason)
        return Retrieval(hits=[_hit("yes", 0.9)])

    report = calibrate(
        index,
        rows,
        ask,
        split={"train": ["t"], "test": ["h1", "h2", "h3"]},
        modes=("rrf", "rerank"),
    )
    assert report["modes"]["rrf"] == {"tau": None}
    entry = report["modes"]["rerank"]
    assert entry["held_out_abstain"] == 1 / 3
    assert entry["held_out_unanswerable_abstain"] == 0.5
    assert isinstance(entry["tau"], float)


def test_calibrate_held_out_without_unanswerable_rows():
    rows = [
        {"id": "t", "kind": "lexical", "must_contain": "yes", "doc_id": "guide.md"},
        {"id": "h", "kind": "lexical", "must_contain": "yes", "doc_id": "guide.md"},
    ]
    index = _Index()

    def ask(_mode, _row):
        return Retrieval(hits=[_hit("yes", 0.6)])

    report = calibrate(index, rows, ask, split={"calibration": ["t"], "holdout": ["h"]}, modes=("cascade",))
    entry = report["modes"]["cascade"]
    assert entry["held_out_unanswerable_abstain"] is None
    assert entry["held_out_abstain"] == 0.0
