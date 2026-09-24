import pytest

from rag.models import Hit, QueryError, Retrieval
from rag.retrieve import (
    _append_fact_siblings,
    _best_child,
    _emit,
    _gate,
    _in_band,
    _parent_scores_from_chunks,
    _prefer_named_year,
    _retain_superseded_siblings,
    _sigmoid,
    agreed_parent,
    fuse,
    retrieve,
)
from rag.telemetry import StageTimer


class Stub:
    def __init__(self, dense=None, bm25=None, parents=None, tau=None):
        self._dense = dense or []
        self._bm25 = bm25 or []
        self._parents = parents or {}
        self._tau = tau
        self.model_id = "test"
        self.model_revision = "rev"

    def dense_search(self, vector, k, doc_id=None):
        return self._dense[:k]

    def bm25_search(self, text, k, doc_id=None):
        return self._bm25[:k]

    def get_parents(self, ids):
        return {parent_id: self._parents[parent_id] for parent_id in ids if parent_id in self._parents}

    def get_tau(self, key):
        return self._tau


def _chunk(chunk_id, parent_id, text="alpha", **extra):
    row = {"chunk_id": chunk_id, "parent_id": parent_id, "embed_text": text}
    row.update(extra)
    return row


def _parent(text="alpha policy"):
    return {
        "text": text,
        "heading_path": "H",
        "source_path": "guide.md",
        "file_sha256": "abc",
        "page_start": 1,
        "page_end": 1,
        "start_char": 0,
        "end_char": len(text),
        "status": "current",
    }


def _hit(parent_id, score=1.0, **extra):
    hit = Hit(parent_id, "text", "H", "a.md", "abc", 1, 1, 0, 4, parent_id, score, False)
    for key, value in extra.items():
        setattr(hit, key, value)
    return hit


def test_fuse_skips_a_repeated_id_and_agreement_bails_on_a_missing_list():
    ranked = fuse([["a", "a", "b"]])
    assert [item[0] for item in ranked] == ["a", "b"]
    assert agreed_parent([], ["a"], ["a"]) is None
    assert agreed_parent([("a", 1.0)], [], ["a"]) is None
    assert agreed_parent([("z", 1.0), ("a", 0.1)], ["a"], ["a"]) is None


def test_gate_keeps_a_single_hit_and_drops_everything_under_tau():
    index = Stub(tau=0.5)
    kept = _gate(Retrieval(hits=[_hit("a", 0.9)]), index, "rrf")
    assert kept.hits[0].confident is True
    empty = _gate(Retrieval(hits=[_hit("a", 0.1)]), index, "rrf")
    assert empty.reason == "no_confident_hit"
    already = _gate(Retrieval(hits=[], reason="no_confident_hit"), index, "rrf")
    assert already.hits == []


def test_band_boundary_is_closed_on_the_low_side():
    assert _in_band(0.85, 1.0) is True
    assert _in_band(0.85 - 1e-9, 1.0) is False


def test_rerank_ignores_a_parent_the_index_does_not_have():
    dense = [_chunk("c1", "gone", dense_score=0.9), _chunk("c2", "p1", dense_score=0.4)]
    index = Stub(dense, [], {"p1": _parent()})
    result = retrieve(
        index,
        "hello",
        lambda _q: [0.0],
        lambda _q, texts: [0.4 for _ in texts],
        mode="rerank",
    )
    assert [hit.parent_id for hit in result.hits] == ["p1"]


def test_sigmoid_clamps_extreme_logits():
    assert _sigmoid(100) == 1.0
    assert _sigmoid(-100) == 0.0
    assert _sigmoid(0) == 0.5


def test_superseded_sibling_is_kept_and_a_stranger_is_not():
    current = _hit("new", version_group="g", superseded=False)
    old = _hit("old", version_group="g", superseded=True)
    other = _hit("other", version_group="elsewhere", superseded=True)
    kept = _retain_superseded_siblings([current], [current, other, old])
    assert [hit.parent_id for hit in kept] == ["new", "old"]
    first = _hit("n1", version_group="g1")
    second = _hit("n2", version_group="g2")
    both = _retain_superseded_siblings(
        [first, second],
        [first, second, _hit("o1", version_group="g1", superseded=True), _hit("o2", version_group="g2", superseded=True)],
    )
    assert {hit.parent_id for hit in both} == {"n1", "n2", "o1", "o2"}


def test_fact_sibling_dedupes_and_trims_to_the_parent_cap(monkeypatch):
    monkeypatch.setattr("rag.retrieve.MAX_PARENTS", 1)
    monkeypatch.setattr(
        "rag.versions.aligned_peer_records",
        lambda index, hits: [
            {},
            {"parent_id": hits[0].parent_id, "text": "same"},
            {"parent_id": "sib", "text": "sibling", "status": "superseded"},
            {"parent_id": "extra", "text": "extra"},
        ],
    )
    hits = _append_fact_siblings(Stub(), "q", [_hit("new", 1.0)])
    assert len(hits) > 1
    assert all(hit.parent_id != "new" for hit in hits)


def test_empty_index_and_rerank_failures():
    stub = Stub()
    with pytest.raises(QueryError, match="empty"):
        retrieve(stub, "   ", lambda _q: [0.0])
    assert retrieve(stub, "hello", lambda _q: [0.0]).hits == []

    dense = [_chunk("c1", "p1", "alpha")]
    lexical = [_chunk("c1", "p1", "alpha")]
    parents = {"p1": _parent()}
    index = Stub(dense, lexical, parents)
    with pytest.raises(QueryError, match="reranker is not available"):
        retrieve(index, "hello", lambda _q: [0.0], None, mode="rerank")
    with pytest.raises(QueryError, match="wrong number"):
        retrieve(index, "hello", lambda _q: [0.0], lambda *_: [0.2, 0.1], mode="rerank")


def test_zero_rerank_window_and_duplicate_parent(monkeypatch):
    monkeypatch.setattr("rag.retrieve.RERANK_K", 0)
    dense = [_chunk("c1", "p1", dense_score=0.4), _chunk("c2", "p1", dense_score=0.9)]
    index = Stub(dense, [], {"p1": _parent()})
    result = retrieve(index, "hello", lambda _q: [0.0], lambda *_: [], mode="rerank")
    assert result.hits == []

    monkeypatch.setattr("rag.retrieve.RERANK_K", 12)
    lexical = [_chunk("c1", "p1", "alpha", bm25_score=2.0), _chunk("c2", "p1", "beta", bm25_score=1.0)]
    scored = Stub([], lexical, {"p1": _parent()})
    ranked = retrieve(scored, "hello", lambda _q: [0.0], lambda q, texts: [0.9 for _ in texts], mode="rerank")
    assert ranked.hits


def test_missing_score_uses_rank_and_named_year_short_circuits():
    order, scores = _parent_scores_from_chunks([_chunk("c", "p")], "missing")
    assert order == ["p"]
    assert scores["p"] == 1.0
    assert _prefer_named_year(["only"], {}, "") == ["only"]
    assert _best_child("p", [], {}) == ""


def test_emit_skips_a_parent_with_no_record_or_child():
    index = Stub(parents={"p1": _parent("kept")})
    result = _emit(
        index,
        ["missing", "p1"],
        {"p1": 0.5, "missing": 0.1},
        {"missing": "c", "p1": "c1"},
        False,
        "query",
    )
    assert result.hits[0].child_id == "c1"


def test_empty_fuse_and_duplicate_chunk_scores(monkeypatch):
    monkeypatch.setattr("rag.retrieve.fuse", lambda *_args, **_kwargs: [])
    dense = [_chunk("c1", "p1", "alpha", dense_score=0.2), _chunk("c2", "p1", "alpha", dense_score=0.1)]
    index = Stub(dense, [], {"p1": _parent()})
    result = retrieve(index, "hello", lambda _q: [0.0], lambda *_: [], mode="cascade")
    assert result.hits == []

    order, scores = _parent_scores_from_chunks(
        [_chunk("c1", "p1", dense_score=0.2), _chunk("c2", "p1", dense_score=0.9), _chunk("c3", "p1", dense_score=0.3)],
        "dense_score",
    )
    assert order == ["p1"]
    assert scores["p1"] == 0.9
    chunks = [_chunk("high", "p"), _chunk("low", "p")]
    assert _best_child("p", chunks, {"high": 0.8, "low": 0.1}) == "high"


def test_cascade_records_a_fuse_top_score():
    dense = [_chunk("c1", "p1", "alpha")]
    lexical = [_chunk("c2", "p2", "beta")]
    parents = {"p1": _parent("alpha"), "p2": _parent("beta")}
    index = Stub(dense, lexical, parents)
    result = retrieve(index, "hello", lambda _q: [0.0], lambda *_: [0.4, 0.3], mode="cascade")
    assert "fuse" in result.stages_ms
    timer = StageTimer()
    from rag.retrieve import _retrieve

    early = _retrieve(Stub(), "hello", lambda _q: [0.0], None, None, "cascade", timer)
    assert early.hits == []
