from rag.evaluate import evaluate, fit_tau, load_rows, row_hit
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
        self.saved = None

    def set_tau(self, key, value):
        self.saved = (key, value)


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
    assert box.saved == (tau_key("test-embed", "rev"), 0.8)
    cascade = next(line for line in lines if line.mode == "cascade" and line.kind == "all")
    assert cascade.recall == 0
    assert cascade.abstain == 1
    lexical = next(line for line in lines if line.mode == "bm25" and line.kind == "lexical")
    assert lexical.recall == 1


def test_golden_file_has_both_kinds():
    rows = load_rows("evals/golden.jsonl")
    assert {row["kind"] for row in rows} == {"lexical", "semantic"}
