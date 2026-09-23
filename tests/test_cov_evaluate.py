"""Branch coverage for rag.evaluate. No model downloads."""

import hashlib

import pytest

import rag.evaluate as evaluate_mod
from rag.evaluate import (
    REQUIRED_KINDS,
    Score,
    _abstain_from_answers,
    _mean_key,
    _pick_gate_line,
    answer_abstained,
    assert_kind_coverage,
    check_gates,
    check_slos,
    conflict_pass,
    evaluate,
    fit_tau,
    format_gate_report,
    format_scores,
    injection_pass,
    kind_coverage,
    load_rows,
    load_split,
    load_suite,
    make_split,
    make_split_from_suite,
    percentile_50,
    pick_live,
    reciprocal,
    row_hit,
    rows_for_split,
    verify_corpus,
)
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


def _line(mode="rrf", kind="all", **overrides):
    fields = {
        "recall": 1.0,
        "mrr": 1.0,
        "abstain": 0.0,
        "n": 1,
        "p50": 1.0,
        "p95": 1.0,
        "p99": 1.0,
    }
    fields.update(overrides)
    return Score(mode=mode, kind=kind, **fields)


class _Index:
    model_id = "m"
    model_revision = "r"

    def __init__(self):
        self.saved = {}

    def set_tau(self, key, value):
        self.saved[key] = value


class _Blank:
    def __bool__(self):
        return True

    def __str__(self):
        return ""


def test_loaders_reject_a_bad_suite_and_keep_blank_lines(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('{"id": "a"}\n\n  \n{"id": "b"}\n', encoding="utf-8")
    assert load_rows(rows_path) == [{"id": "a"}, {"id": "b"}]
    split_path = tmp_path / "split.json"
    split_path.write_text('{"train": ["a"]}\n', encoding="utf-8")
    assert load_split(split_path)["train"] == ["a"]

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    listed = tmp_path / "list.yaml"
    listed.write_text("- a\n", encoding="utf-8")
    no_gates = tmp_path / "no-gates.yaml"
    no_gates.write_text("corpus: {}\n", encoding="utf-8")
    no_corpus = tmp_path / "no-corpus.yaml"
    no_corpus.write_text("gates: {}\n", encoding="utf-8")
    ok = tmp_path / "ok.yaml"
    ok.write_text("gates: {}\ncorpus: {}\n", encoding="utf-8")
    for path in (empty, listed, no_gates, no_corpus):
        with pytest.raises(ValueError, match="suite missing"):
            load_suite(path)
    assert load_suite(ok)["gates"] == {}


def test_rows_for_split_aliases_and_misses():
    rows = [{"id": "a"}, {"id": "b"}]
    assert rows_for_split(rows, {"train": ["a"]}, "train") == [rows[0]]
    assert rows_for_split(rows, {"calibration": ["b"]}, "train") == [rows[1]]
    assert rows_for_split(rows, {"other": ["a"]}, "train") == []
    assert rows_for_split(rows, {"custom": ["a"]}, "custom") == [rows[0]]
    assert rows_for_split(rows, {}, "custom") == []
    assert rows_for_split(rows, {"holdout": ["a"]}, "test") == [rows[0]]


def test_make_split_sizes_singleton_buckets_and_an_empty_one(monkeypatch):
    rows = [
        {"id": "b", "kind": "k"},
        {"id": "a", "kind": "k"},
        {"id": "solo", "kind": "z"},
    ]
    low = make_split(rows, calibration_fraction=0.0, seed=1)
    assert len(low["train"]) == 1
    assert len(low["test"]) == 2
    assert "solo" in low["test"]
    assert low["calibration"] == low["train"]
    assert low["holdout"] == low["test"]
    high = make_split(rows, calibration_fraction=1.0, seed=1)
    # Pair clamps to len-1. A singleton with fraction >= 0.5 stays in train.
    assert len(high["train"]) == 2
    assert high["test"] != []
    assert "solo" in high["train"]
    unknown = make_split([{"id": "z"}], stratify_by="missing", calibration_fraction=0.2)
    assert unknown["train"] == []
    assert unknown["test"] == ["z"]

    class _WithEmpty(evaluate_mod.defaultdict):
        def __init__(self, default_factory=None, **kwargs):
            super().__init__(default_factory, **kwargs)
            self["__empty__"] = []

    monkeypatch.setattr(evaluate_mod, "defaultdict", _WithEmpty)
    emptied = make_split([{"id": "a", "kind": "k"}], calibration_fraction=0.2)
    assert emptied["train"] == []
    assert emptied["test"] == ["a"]


def test_make_split_from_suite_reads_optional_blocks():
    rows = [{"id": "a", "kind": "lexical"}, {"id": "b", "kind": "lexical"}]
    bare = make_split_from_suite(rows, {})
    assert set(bare["train"]) | set(bare["test"]) == {"a", "b"}
    noted = make_split_from_suite(rows, {"splits": {"note": "x"}})
    assert set(noted["calibration"]) | set(noted["holdout"]) == {"a", "b"}
    custom = make_split_from_suite(
        rows,
        {"splits": {"calibration": {"seed": 3, "fraction": 0.5, "stratify_by": "kind"}}},
    )
    assert set(custom["train"]) | set(custom["test"]) == {"a", "b"}


def test_verify_corpus_matches_mismatches_and_missing_files(tmp_path):
    assert verify_corpus({}) == []
    payload = b"pinned"
    doc = tmp_path / "sub" / "doc.txt"
    doc.parent.mkdir()
    doc.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    failures = verify_corpus(
        {
            "corpus": {
                "root": "ignored",
                "files": [
                    {"path": "sub/doc.txt", "sha256": digest},
                    {"path": "sub/doc.txt", "sha256": "deadbeef"},
                    {"path": "missing.txt"},
                    {"path": "nope.txt", "sha256": digest},
                ],
            }
        },
        root=tmp_path,
    )
    assert len(failures) == 3
    assert failures[0].startswith("sha256 mismatch sub/doc.txt")
    assert failures[1].startswith("missing")
    assert failures[2].startswith("missing")
    bare = verify_corpus({"corpus": {"files": [{"path": "missing-cov.txt", "sha256": "abc"}]}})
    assert bare and bare[0].startswith("missing")
    rooted = verify_corpus(
        {"corpus": {"root": str(tmp_path), "files": [{"path": "sub/doc.txt", "sha256": digest}]}}
    )
    assert rooted == []


def test_conflict_and_injection_pass_edges():
    assert conflict_pass("", {}) is False
    assert conflict_pass("x", {"expect": "y"}) is False
    assert conflict_pass("y", {"expect": "y", "superseded_value": "old"}) is False
    assert conflict_pass("y old", {"current_value": "y", "superseded_value": "old"}) is False
    assert conflict_pass("y old superseded", {"current_value": "y", "superseded_value": "old"})
    assert conflict_pass(
        "y old see old.pdf",
        {"current_value": "y", "superseded_value": "old", "superseded_doc": "Old.PDF"},
    )
    assert (
        conflict_pass(
            "y old",
            {"current_value": "y", "superseded_value": "old", "superseded_doc": "missing.pdf"},
        )
        is False
    )
    assert conflict_pass("y superseded", {"current_value": "y"})

    assert injection_pass("", {"expect": "x"}) is False
    assert injection_pass("COMPROMISED", {}) is False
    assert injection_pass("plain", {}) is True
    assert injection_pass("plain", {"expect": "secret"}) is False
    assert injection_pass("see secret", {"must_contain": "secret"}) is True
    assert injection_pass("ok", {"must_not_contain": "NOPE", "expect": "ok"}) is True
    assert injection_pass("text", {"must_not_contain": _Blank()}) is True


def test_kind_coverage_fit_tau_and_hit_edges():
    assert "missing kind lexical" in assert_kind_coverage([])
    assert any("unanswerable" in item for item in assert_kind_coverage([]))
    assert any("conflict rows" in item for item in assert_kind_coverage([]))
    rows = [{"kind": kind} for kind in REQUIRED_KINDS]
    rows += [{"kind": "unanswerable"}] * 24
    rows += [{"kind": "conflict"}] * 7
    assert assert_kind_coverage(rows) == []
    assert kind_coverage([{}]) == {"?": 1}
    assert kind_coverage([{"kind": "lexical"}]) == {"lexical": 1}
    assert fit_tau([]) is None
    assert fit_tau([0.2, 0.8]) == 0.2
    assert percentile_50([1.0, 2.0, 3.0]) == 2.0

    assert row_hit([_hit("a")], {}) is False
    assert row_hit([_hit("a")], {"must_contain": "a"}) is False
    assert reciprocal([], {"doc_id": "guide.md"}) == 0.0
    assert reciprocal([], {}) == 0.0
    hit = _hit("NEEDLE", 0.5)
    row = {"must_contain": "NEEDLE", "doc_id": "guide.md"}
    assert row_hit([hit], row) is True
    assert reciprocal([hit], row) == 1.0
    assert reciprocal([_hit("no", 1), hit], row) == 0.5
    assert row_hit([_hit("NEEDLE", 1, doc_id="other.md")], row) is False
    assert row_hit([_hit("nope", 1)], row) is False
    assert reciprocal([_hit("nope", 1)], row) == 0.0
    assert row_hit([_hit("nope", 1), hit], row, k=1) is False


def test_abstain_helpers_and_pick_live():
    assert answer_abstained("")
    assert answer_abstained("Do not say that.")
    assert not answer_abstained("India baseline was 14,644.")
    assert _abstain_from_answers("unanswerable_abstention", [], {}) is None
    assert (
        _abstain_from_answers(
            "unanswerable_abstention",
            [{"id": "u", "kind": "unanswerable"}],
            {},
        )
        is None
    )
    rows = [
        {"id": "u", "kind": "unanswerable"},
        {"id": "a", "kind": "lexical", "must_contain": "z"},
    ]
    assert _abstain_from_answers("unanswerable_abstention", rows, {"u": ""}) == 1
    assert _abstain_from_answers("answerable_abstention", rows, {"a": "z is here"}) == 0
    assert _abstain_from_answers("answerable_abstention", [{"id": "u", "kind": "unanswerable"}], {}) is None
    assert _abstain_from_answers("answerable_abstention", rows, {}) is None

    assert pick_live([]) == "cascade"
    assert pick_live([_line(kind="lexical")]) == "cascade"
    assert (
        pick_live(
            [
                _line(mode="weird", kind="holdout", recall=0.1, mrr=0.4, p50=3),
                _line(mode="bm25", kind="holdout", recall=0.2, mrr=0.9, p50=9),
                _line(kind="lexical", recall=0.99, mrr=0.99),
            ]
        )
        == "bm25"
    )
    assert (
        pick_live(
            [
                _line(mode="weird", kind="all", recall=0.99, mrr=0.5, p50=1),
                _line(mode="rrf", kind="all", recall=0.99, mrr=0.5, p50=1),
                _line(kind="lexical", recall=0.1, mrr=0.1),
            ]
        )
        == "rrf"
    )
    assert (
        pick_live(
            [
                _line(mode="bm25", kind="all", recall=0.5, mrr=0.99, p50=1),
                _line(mode="cascade", kind="all", recall=0.95, mrr=0.2, p50=50),
            ]
        )
        == "cascade"
    )
    assert (
        pick_live(
            [
                _line(mode="rerank", kind="holdout", recall=0.95, mrr=0.8, p50=20),
                _line(mode="rrf", kind="holdout", recall=0.95, mrr=0.8, p50=5),
            ]
        )
        == "rrf"
    )


def test_format_scores_and_mean_key():
    lines = [
        _line(
            answerable_abstain=0.1,
            unanswerable_abstain=0.2,
            groundedness=0.3,
            citation_precision=0.4,
            citation_recall=0.5,
        ),
        _line(kind="lexical"),
        _line(answerable_abstain=None, groundedness=0.3, citation_precision=None, citation_recall=0.2),
        _line(groundedness=0.3, citation_precision=0.4, citation_recall=None),
        _line(groundedness=None, citation_precision=0.4, citation_recall=0.2),
    ]
    text = format_scores(lines, 0.25)
    assert "tau=0.2500" in text
    assert "ans_abs=0.10" in text
    assert "ground=0.30" in text
    assert format_scores([], None).splitlines()[-2] == "tau unset"
    assert _mean_key(None, "a") is None
    assert _mean_key([], "a") is None
    assert _mean_key([{"a": None}, {}], "a") is None
    assert _mean_key([{"a": 1.0}, {"a": 3.0}, {"a": None}], "a") == 2.0


def test_pick_gate_line_and_slos():
    assert _pick_gate_line([]) is None
    assert _pick_gate_line([_line(kind="holdout")]) is None
    assert _pick_gate_line([_line(mode="weird")]).mode == "weird"
    assert _pick_gate_line([_line(mode="bm25")]).mode == "bm25"
    assert _pick_gate_line([_line(mode="dense")], "dense").mode == "dense"
    assert _pick_gate_line([_line(mode="rrf")], "missing").mode == "rrf"

    lines = [
        _line(mode="dense", p95=999),
        _line(mode="rrf", kind="holdout", p95=999),
        _line(mode="rrf", p95=10),
        _line(mode="cascade", p95=400),
        _line(mode="rerank", p95=500),
        _line(mode="bm25", p95=500),
    ]
    assert check_slos(lines, ttft_p95=50) == ["retrieve p95 500ms exceeds 400ms (rerank)"]
    assert check_slos(lines, ttft_p95=1200) == ["retrieve p95 500ms exceeds 400ms (rerank)"]
    assert check_slos(
        [
            _line(mode="dense", p95=999),
            _line(mode="bm25", kind="holdout", p95=500),
            _line(mode="bm25", p95=500),
        ],
        live_mode="bm25",
        ttft_p95=5000,
        suite={"slo": {"retrieve_p95_ms": 10, "ttft_p95_ms": 9000}},
    ) == ["retrieve p95 500ms exceeds 10ms (bm25)"]
    assert check_slos(
        [_line(mode="rrf", p95=10)],
        ttft_p95=2000,
        suite={"slo": {"retrieve_p95_ms": 50, "ttft_p95_ms": 100}},
    ) == ["ttft p95 2000ms exceeds 100ms"]
    assert check_slos([], ttft_p95=None) == []
    assert check_slos([_line(p95=10)], suite={}) == []
    assert check_slos([_line(p95=10)], suite={"slo": {}}) == []
    assert check_slos([_line(p95=10)], suite={"gates": {}}) == []


def test_check_gates_metrics_holdout_and_answers():
    line = _line(
        recall=0.5,
        mrr=0.4,
        groundedness=None,
        citation_precision=0.9,
        citation_recall=0.95,
        answerable_abstain=0.05,
        unanswerable_abstain=0.8,
        p95=10,
    )
    suite = {
        "gates": {
            "recall_at_5": {"min": 0.9},
            "mrr": {"min": 0.1},
            "groundedness": {"min": 0.5},
            "citation_precision": {"max": 0.5},
            "citation_recall": {"min": 0.1, "max": 0.99},
            "answerable_abstention": {"max": 0.2},
            "unanswerable_abstention": {"min": 0.2},
            "conflict_disclosure": {"min": 1},
            "injection_resisted": {"min": 1},
            "recall_note": "skip",
            "mystery": {"min": 0.1},
        }
    }
    assert check_gates(suite, [line]) == [
        "recall_at_5=0.500 < 0.900",
        "citation_precision=0.900 > 0.500",
    ]
    assert check_gates({"gates": {"recall_at_5": {"min": 0.9, "scope": "all"}}}, [_line(recall=0.5, p95=1)]) == [
        "recall_at_5=0.500 < 0.900 (scope=all)"
    ]
    assert check_gates({"gates": {"mrr": {"min": 0.1, "scope": "holdout"}}}, [_line()]) == [
        "mrr: holdout scores missing"
    ]
    assert check_gates(
        {"gates": {"mrr": {"min": 0.1, "scope": "holdout"}}},
        [],
        live_mode="rrf",
    ) == ["mrr: holdout scores missing"]
    assert check_gates(
        {"gates": {"mrr": {"min": 0.8, "scope": "holdout"}}},
        [_line(mrr=0.2, p95=1), _line(kind="holdout", mrr=0.2, p95=1)],
    ) == ["mrr=0.200 < 0.800 (scope=holdout)"]
    assert check_gates({}, [_line()]) == []
    assert check_gates({"gates": {"recall_at_5": {"min": 0.5}}}, [], live_mode="bm25") == []
    assert format_gate_report([]) == "suite gates: PASS"
    assert format_gate_report(["a", "b"]) == "suite gates: FAIL\n- a\n- b"

    rows = [
        {"id": "u1", "kind": "unanswerable"},
        {"id": "u2", "kind": "unanswerable"},
    ]
    answers = {"u1": "not in the documents", "u2": "a concrete answer"}
    hold_lines = [_line(unanswerable_abstain=0.0, p95=1), _line(kind="holdout", unanswerable_abstain=0.0, p95=1)]
    assert (
        check_gates(
            {"gates": {"unanswerable_abstention": {"min": 0.9, "scope": "holdout"}}},
            hold_lines,
            answers=answers,
            rows=rows,
            split={"holdout": ["u1"]},
        )
        == []
    )
    assert (
        check_gates(
            {"gates": {"unanswerable_abstention": {"min": 0.9, "scope": "holdout"}}},
            hold_lines,
            answers={"u1": "not stated"},
            rows=[{"id": "u1", "kind": "unanswerable"}],
            split=None,
        )
        == []
    )
    assert check_gates(
        {"gates": {"unanswerable_abstention": {"min": 0.5}}},
        [_line(unanswerable_abstain=0.0, p95=1)],
        answers={},
        rows=[{"id": "u1", "kind": "unanswerable"}],
    ) == ["unanswerable_abstention=0.000 < 0.500"]
    assert (
        check_gates(
            {"gates": {"answerable_abstention": {"max": 0.1}}},
            [_line(answerable_abstain=0.9, p95=1)],
            answers={"a": "x is here"},
            rows=[{"id": "a", "kind": "lexical", "must_contain": "x"}],
        )
        == []
    )
    assert check_gates(
        {"gates": {"answerable_abstention": {"max": 0.0}}},
        [_line(answerable_abstain=0.5, p95=1)],
        answers={"a": "text"},
        rows=None,
    ) == ["answerable_abstention=0.500 > 0.000"]
    assert (
        check_gates(
            {"gates": {"recall_at_5": {"min": 0.1}}},
            [_line(recall=1, p95=1)],
            answers={},
            rows=[],
        )
        == []
    )


def test_check_gates_conflict_injection_and_regression():
    quiet = _line(p95=1)
    assert check_gates(
        {"gates": {"conflict_disclosure": {"min": 1}, "injection_resisted": {"min": 1}}},
        [quiet],
        answers={},
        rows=[],
    ) == [
        "conflict_disclosure: no conflict rows",
        "injection_resisted: no injection rows",
    ]
    conflict_rows = [{"id": "c", "kind": "conflict", "current_value": "2040", "superseded_value": "2050"}]
    assert check_gates(
        {"gates": {"conflict_disclosure": {"on": True}}},
        [quiet],
        answers={"c": "2040 only"},
        rows=conflict_rows,
    ) == ["conflict_disclosure=0.000 < 1.000 (0/1)"]
    assert (
        check_gates(
            {"gates": {"conflict_disclosure": {"min": 0}}},
            [quiet],
            answers={"c": "2040 only"},
            rows=conflict_rows,
        )
        == []
    )
    assert (
        check_gates(
            {"gates": {"conflict_disclosure": {"min": 1}}},
            [quiet],
            answers={"c": "2040 superseded 2050"},
            rows=conflict_rows,
        )
        == []
    )
    inj_rows = [{"id": "i", "kind": "injection", "must_not_contain": "COMPROMISED", "expect": "safe"}]
    assert check_gates(
        {"gates": {"injection_resisted": {"on": True}}},
        [quiet],
        answers={"i": "COMPROMISED"},
        rows=inj_rows,
    ) == ["injection_resisted=0.000 < 1.000 (0/1)"]
    assert (
        check_gates(
            {"gates": {"injection_resisted": {"min": 1}}},
            [quiet],
            answers={"i": "safe"},
            rows=inj_rows,
        )
        == []
    )

    failing = _line(recall=0.5, p95=200)
    assert check_gates(
        {
            "gates": {"recall_at_5": {"min": 0.1}},
            "regression": {"max_recall_drop": 0.03, "max_p95_increase_pct": 15},
        },
        [failing],
        baseline={"live_mode": "rrf", "scores": [{"mode": "rrf", "kind": "all", "recall": 0.9, "p95": 100}]},
    ) == [
        "recall drop 0.400 exceeds 0.030",
        "p95 increase 100.0% exceeds 15.0%",
    ]
    assert (
        check_gates(
            {"regression": {"max_recall_drop": 0.5, "max_p95_increase_pct": 50}},
            [_line(recall=0.9, p95=110)],
            baseline={"live_mode": "rrf", "scores": [{"mode": "rrf", "kind": "all", "recall": 0.9, "p95": 100}]},
            live_mode="missing",
        )
        == []
    )
    assert (
        check_gates(
            {"regression": {"max_recall_drop": 0.03, "max_p95_increase_pct": 15}},
            [_line(recall=0.9, p95=100)],
            baseline={"scores": [{"mode": "rrf", "kind": "all", "recall": 0.9, "p95": 100}]},
        )
        == []
    )
    assert check_gates(
        {"regression": {"max_p95_increase_pct": 15}},
        [_line(recall=0.9, p95=200)],
        baseline={"scores": [{"mode": "rrf", "kind": "all", "p95": 100}]},
    ) == ["p95 increase 100.0% exceeds 15.0%"]
    assert (
        check_gates(
            {"regression": {"on": True}},
            [_line(recall=0.9, p95=200)],
            baseline={"scores": [{"mode": "rrf", "kind": "all", "recall": 0.9, "p95": None}]},
        )
        == []
    )
    assert (
        check_gates(
            {"regression": {"on": True}},
            [_line(recall=0.9, p95=0)],
            baseline={"scores": [{"mode": "rrf", "kind": "all", "recall": 0.9, "p95": 100}]},
        )
        == []
    )
    assert (
        check_gates(
            {"regression": {"on": True}},
            [_line(recall=0.9, p95=50)],
            baseline={"scores": [{"mode": "rrf", "kind": "all", "recall": 0.9, "p95": 0}]},
        )
        == []
    )
    assert check_gates({"regression": {"on": True}}, [_line()], baseline=None) == []
    assert check_gates({"regression": {"on": True}}, [], baseline={"scores": []}, live_mode="rrf") == []
    assert (
        check_gates(
            {"regression": {"on": True}},
            [_line(p95=1)],
            baseline={"live_mode": "nope"},
            live_mode="missing",
        )
        == []
    )


def test_evaluate_scores_holdout_and_empty_rows():
    rows = [
        {"id": "a", "kind": "lexical", "must_contain": "NEEDLE", "doc_id": "guide.md"},
        {"id": "rank2", "kind": "semantic", "must_contain": "NEEDLE", "doc_id": "guide.md"},
        {"id": "unans", "kind": "unanswerable"},
        {"id": "outside", "kind": "lexical", "must_contain": "NEEDLE", "doc_id": "guide.md"},
    ]

    def ask(mode, row):
        if mode == "dense":
            reason = "no_confident_hit" if row["id"] == "unans" else ""
            return Retrieval(hits=[], reason=reason)
        if mode == "bm25":
            return Retrieval(hits=[_hit("nope", 0.3)], stages_ms={"embed": 4.0})
        if mode == "cascade" and row["id"] == "rank2":
            return Retrieval(
                hits=[_hit("zzz", 0.9), _hit("NEEDLE", 0.4)],
                stages_ms={"total": 25.0},
            )
        if row["id"] == "unans":
            return Retrieval(hits=[_hit("x", 0.1)], reason="no_confident_hit", stages_ms={})
        return Retrieval(hits=[_hit("NEEDLE", 0.8)], stages_ms={"total": 12.0})

    def verify(row, _result):
        if row["id"] == "a":
            return {"groundedness": 1.0, "citation_precision": 1.0, "citation_recall": 1.0}
        return {"groundedness": 0.0, "citation_precision": None, "citation_recall": 0.0}

    index = _Index()
    lines, fitted = evaluate(index, rows, ask, verify, split={"holdout": ["a", "rank2"]})
    assert fitted == 0.8
    overall = next(line for line in lines if line.mode == "rrf" and line.kind == "all")
    assert overall.groundedness == 0.25
    assert overall.citation_precision == 1.0
    assert overall.answerable_abstain == 0.0
    assert overall.unanswerable_abstain == 1.0
    lexical = next(line for line in lines if line.mode == "rrf" and line.kind == "lexical")
    assert lexical.groundedness is None
    assert lexical.answerable_abstain is None
    hold = next(line for line in lines if line.mode == "cascade" and line.kind == "holdout")
    assert hold.n == 2
    assert hold.mrr == 0.75
    assert hold.unanswerable_abstain is None
    assert "tau=0.8000" in format_scores(lines, fitted)
    assert any("rerank" in key for key in index.saved)

    def ask_hits(_mode, _row):
        return Retrieval(hits=[_hit("x", 0.2)], reason="no_confident_hit", stages_ms={"total": 8.0})

    blank_index = _Index()
    blank_lines, blank_fitted = evaluate(
        blank_index,
        [{"id": "u", "kind": "unanswerable"}],
        ask_hits,
        split={"holdout": ["u"]},
    )
    assert blank_fitted is None
    assert blank_index.saved == {}
    blank_all = next(line for line in blank_lines if line.kind == "all" and line.mode == "bm25")
    assert blank_all.recall == 0.0
    assert blank_all.answerable_abstain is None
    assert blank_all.unanswerable_abstain == 1.0
    blank_hold = next(line for line in blank_lines if line.kind == "holdout" and line.mode == "bm25")
    assert blank_hold.recall == 0.0
    assert blank_hold.answerable_abstain is None
    assert blank_hold.unanswerable_abstain == 1.0

    def ask_empty(_mode, _row):
        return Retrieval(hits=[], stages_ms={"total": 5.0})

    only, only_fitted = evaluate(
        _Index(),
        [{"id": "a", "kind": "lexical", "must_contain": "NEEDLE", "doc_id": "guide.md"}],
        ask_empty,
    )
    assert only_fitted is None
    assert all(line.kind != "holdout" for line in only)
    only_all = next(line for line in only if line.kind == "all")
    assert only_all.unanswerable_abstain is None
    assert only_all.answerable_abstain == 0.0
    assert only_all.recall == 0.0

    def ask_unused(_mode, _row):
        raise AssertionError("empty rows do not ask")

    empty_lines, empty_fitted = evaluate(_Index(), [], ask_unused, split={"holdout": ["missing"]})
    assert empty_lines == []
    assert empty_fitted is None

