from rag.evaluate import evaluate, fit_tau, load_rows, load_split, percentile, row_hit, rows_for_split
from rag.models import Hit, Retrieval, tau_key


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


def test_fit_tau_drops_only_the_bottom_tail():
    scores = list(range(20))
    tau = fit_tau(scores)
    assert tau == 1
    assert sum(score < tau for score in scores) == 1
    assert sum(score >= tau for score in scores) == 19


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
    rows = load_rows("evals/fixture.jsonl")
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
    rows = load_rows("evals/policy.jsonl")
    kinds = {row["kind"] for row in rows}
    assert {"lexical", "semantic", "multi-hop", "unanswerable", "adversarial"} <= kinds
    assert len(rows) >= 120
    assert all(row.get("id") for row in rows)
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_split_file_covers_every_policy_row():
    rows = load_rows("evals/policy.jsonl")
    split = load_split("evals/split.json")
    train = set(split["train"])
    test = set(split["test"])
    ids = {row["id"] for row in rows}
    assert train.isdisjoint(test)
    assert train | test == ids
    assert len(test) >= 25
    held = rows_for_split(rows, split, "test")
    assert {row["id"] for row in held} == test


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


def test_pick_live_keeps_the_fast_mode_that_still_recalls():
    from rag.evaluate import Score, pick_live

    def line(mode, recall, p50, mrr=0.5):
        return Score(mode, "all", recall, mrr, 0.0, 4, p50)

    assert pick_live([line("cascade", 0.9, 10), line("rerank", 0.92, 40)]) == "cascade"
    assert pick_live([line("cascade", 0.8, 10), line("rerank", 0.92, 40)]) == "rerank"
    assert pick_live([line("cascade", 0.9, 40), line("rerank", 0.9, 40)]) == "rerank"
    assert pick_live([
        line("dense", 1, 240, 0.77),
        line("rrf", 1, 250, 0.93),
        line("rerank", 1, 5000, 0.94),
        line("cascade", 0.88, 5000, 0.88),
        line("bm25", 0.94, 1, 0.81),
    ]) == "rrf"
