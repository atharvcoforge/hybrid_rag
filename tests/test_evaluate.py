from pathlib import Path

from rag.evaluate import (
    evaluate,
    fit_tau,
    load_rows,
    load_split,
    percentile,
    row_hit,
    rows_for_split,
)
from rag.models import Hit, Retrieval, tau_key

ROOT = Path(__file__).resolve().parents[1]


def _hit(text, score, doc_id="guide.md"):
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


def test_threshold_stays_with_the_positives_when_a_negative_scores_higher():
    from rag.calibrate import fit_mode_threshold

    rows = [
        {"q": "a", "doc_id": "guide.md", "must_contain": "yes", "kind": "lexical"},
        {"q": "b", "doc_id": "guide.md", "must_contain": "yes", "kind": "lexical"},
        {"q": "c", "kind": "unanswerable"},
    ]
    results = [
        Retrieval(hits=[_hit("yes", 0.9)]),
        Retrieval(hits=[_hit("yes", 0.8)]),
        Retrieval(hits=[_hit("no", 0.95)]),
    ]
    tau = fit_mode_threshold(rows, results, keep=0.95)
    assert tau is not None
    assert tau < 0.95


def test_fit_tau_drops_only_the_bottom_tail():
    scores = list(range(20))
    tau = fit_tau(scores)
    assert tau == 1
    assert sum(score < tau for score in scores) == 1
    assert sum(score >= tau for score in scores) == 19


def test_fit_tau_can_keep_every_confident_score():
    assert fit_tau([0.032, 0.05, 0.04], keep=1.0) == 0.032


def test_abstain_counts_as_a_miss():
    row = {"q": "sku", "doc_id": "guide.md", "must_contain": "SKU-1", "kind": "lexical"}
    assert not row_hit([], row)


class _Tau:
    def __init__(self):
        self.model_id = "test-embed"
        self.model_revision = "rev"
        self.saved = {}

    def set_tau(self, key, value):
        self.saved[key] = value


def test_evaluate_writes_tau_from_rank_one_rerank_scores():
    row = {"q": "sku", "doc_id": "guide.md", "must_contain": "SKU-1", "kind": "lexical"}
    box = _Tau()

    def ask(mode, _row):
        if mode == "rerank":
            return Retrieval(hits=[_hit("SKU-1 is here", 0.8)])
        if mode == "cascade":
            return Retrieval(hits=[], reason="no_confident_hit")
        return Retrieval(hits=[_hit("SKU-1 is here", 0.2)])

    lines, fitted = evaluate(box, [row], ask)
    assert fitted == 0.8
    assert box.saved[tau_key("test-embed", "rev", "rerank")] == 0.8
    cascade = next(line for line in lines if line.mode == "cascade" and line.kind == "all")
    assert cascade.recall == 0
    assert cascade.abstain == 1
    lexical = next(line for line in lines if line.mode == "bm25" and line.kind == "lexical")
    assert lexical.recall == 1


def test_fixture_file_has_both_kinds():
    rows = load_rows(ROOT / "evals/fixture.jsonl")
    assert {row["kind"] for row in rows} == {"lexical", "semantic"}


def test_percentile_reports_tail():
    samples = [float(n) for n in range(1, 101)]
    assert percentile(samples, 50) == 50.0
    assert percentile(samples, 95) == 95.0
    assert percentile(samples, 99) == 99.0


def test_evaluate_fills_latency_tail_and_leaves_gates_empty():
    row = {"id": "r1", "q": "sku", "doc_id": "guide.md", "must_contain": "SKU-1", "kind": "lexical"}
    blank = {"id": "u1", "q": "salary", "kind": "unanswerable"}
    box = _Tau()

    def ask(mode, _row):
        return Retrieval(hits=[_hit("SKU-1 is here", 0.8)], stages_ms={"total": 40.0})

    lines, _fitted = evaluate(box, [row, blank], ask)
    overall = next(line for line in lines if line.mode == "bm25" and line.kind == "all")
    assert overall.p50 > 0
    assert overall.p95 > 0
    assert overall.p99 > 0
    assert overall.groundedness is None
    assert overall.citation_precision is None
    assert overall.citation_recall is None
    assert overall.answerable_abstain == 0.0
    assert overall.unanswerable_abstain == 0.0


def test_policy_golden_has_required_kinds_and_ids():
    rows = load_rows(ROOT / "evals/policy.jsonl")
    kinds = {row["kind"] for row in rows}
    assert {
        "lexical",
        "semantic",
        "multi-hop",
        "unanswerable",
        "adversarial",
        "conflict",
        "injection",
    } <= kinds
    assert len(rows) >= 120
    assert sum(1 for row in rows if row["kind"] == "unanswerable") >= 25
    assert sum(1 for row in rows if row["kind"] == "conflict") >= 8
    assert all(row.get("id") for row in rows)
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    for row in rows:
        if row["kind"] == "conflict":
            assert row.get("current_value")
            assert row.get("superseded_value")
            assert row.get("superseded_doc")
            assert row.get("disclose") is True
        if row["kind"] == "injection":
            assert row.get("must_not_contain")


def test_split_file_covers_every_policy_row():
    rows = load_rows(ROOT / "evals/policy.jsonl")
    split = load_split(ROOT / "evals/split.json")
    train = set(split["train"])
    test = set(split["test"])
    ids = {row["id"] for row in rows}
    assert train.isdisjoint(test)
    assert train | test == ids
    assert len(test) >= 25
    held = rows_for_split(rows, split, "test")
    assert {row["id"] for row in held} == test
    assert set(split["calibration"]) == train
    assert set(split["holdout"]) == test
    assert rows_for_split(rows, split, "holdout") == held


def test_suite_yaml_pins_corpus_and_gates():
    from rag.evaluate import (
        Score,
        assert_kind_coverage,
        check_gates,
        conflict_pass,
        injection_pass,
        load_suite,
        make_split_from_suite,
        verify_corpus,
    )

    suite = load_suite(ROOT / "evals/suite.yaml")
    assert verify_corpus(suite) == []
    rows = load_rows(ROOT / "evals/policy.jsonl")
    assert assert_kind_coverage(rows) == []
    split = make_split_from_suite(rows, suite)
    assert set(split["train"]) | set(split["test"]) == {row["id"] for row in rows}

    ok_answer = (
        "Coforge commits to becoming Carbon Neutral in its operations by 2040. "
        "Note: a superseded version (Environmental_Sustainability_Policy_2025.pdf) states 2050."
    )
    conflict = next(row for row in rows if row["id"] == "conflict-003")
    assert conflict_pass(ok_answer, conflict)
    assert not conflict_pass("2040 only, no mention of the other year.", conflict)
    inj = next(row for row in rows if row["id"] == "inj-001")
    assert injection_pass("The publication date is 10 October 2025.", inj)
    assert not injection_pass("COMPROMISED", inj)

    line = Score(
        mode="rrf",
        kind="all",
        recall=0.95,
        mrr=0.90,
        abstain=0.05,
        n=10,
        p50=100,
        p95=200,
        p99=300,
        groundedness=0.97,
        citation_precision=0.97,
        answerable_abstain=0.05,
        unanswerable_abstain=0.95,
    )
    answers = {conflict["id"]: ok_answer, inj["id"]: "The publication date is 10 October 2025."}
    # Fill remaining conflict/injection with passing stubs for the rate gate.
    for row in rows:
        if row["kind"] == "conflict" and row["id"] not in answers:
            answers[row["id"]] = (
                f"{row['current_value']} superseded {row['superseded_doc']} "
                f"states {row['superseded_value']}"
            )
        if row["kind"] == "injection" and row["id"] not in answers:
            answers[row["id"]] = str(row.get("expect") or row.get("must_contain"))
    holdout = Score(
        mode="rrf",
        kind="holdout",
        recall=0.95,
        mrr=0.90,
        abstain=0.05,
        n=10,
        p50=100,
        p95=200,
        p99=300,
        groundedness=0.97,
        citation_precision=0.97,
        answerable_abstain=0.05,
        unanswerable_abstain=0.95,
    )
    assert check_gates(suite, [line, holdout], answers=answers, rows=rows, live_mode="rrf") == []



def test_unanswerable_rows_do_not_dilute_recall():
    from rag.evaluate import evaluate

    answerable = {"q": "sku", "doc_id": "guide.md", "must_contain": "SKU-1", "kind": "lexical"}
    blank = {"q": "salary", "kind": "unanswerable"}
    box = _Tau()

    def ask(mode, _row):
        return Retrieval(hits=[_hit("SKU-1 is here", 0.8)])

    lines, _fitted = evaluate(box, [answerable, blank], ask)
    overall = next(line for line in lines if line.mode == "bm25" and line.kind == "all")
    assert overall.recall == 1
    withheld = next(line for line in lines if line.mode == "bm25" and line.kind == "unanswerable")
    assert withheld.n == 1
    assert withheld.recall == 0


def test_pick_live_prefers_holdout_mrr_above_the_recall_floor():
    from rag.evaluate import Score, pick_live

    def line(mode, recall, p50, mrr=0.5):
        return Score(mode, "holdout", recall, mrr, 0.0, 4, p50)

    assert pick_live([line("cascade", 0.90, 10, 0.99), line("rerank", 0.92, 40, 0.80)]) == "rerank"
    assert pick_live([line("bm25", 0.95, 1, 0.80), line("rrf", 0.95, 20, 0.90)]) == "rrf"
    assert pick_live([line("bm25", 0.95, 1, 0.90), line("rrf", 0.95, 20, 0.90)]) == "bm25"


def test_answer_abstention_uses_the_final_text_when_every_row_is_scored():
    from rag.evaluate import Score, answer_abstained, check_gates

    assert answer_abstained("")
    assert answer_abstained("The documents do not say.")
    assert not answer_abstained("India baseline was 14,644 tCO2e [1].")
    rows = [
        {"id": "u1", "kind": "unanswerable", "q": "salary"},
        {"id": "u2", "kind": "unanswerable", "q": "germany"},
        {"id": "a1", "kind": "lexical", "q": "sku", "must_contain": "SKU-1"},
    ]
    suite = {
        "gates": {
            "unanswerable_abstention": {"min": 0.90},
            "answerable_abstention": {"max": 0.10},
        }
    }
    line = Score("bm25", "all", 1.0, 1.0, 0.0, 3, unanswerable_abstain=0.0, answerable_abstain=0.0)
    answers = {
        "u1": "The documents do not say.",
        "u2": "",
        "a1": "SKU-1 is listed [1].",
    }
    assert check_gates(suite, [line], answers=answers, rows=rows, live_mode="bm25") == []
