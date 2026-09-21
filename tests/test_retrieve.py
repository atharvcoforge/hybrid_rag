import pytest

from rag.models import QueryError
from rag.retrieve import agreed_parent, fuse, retrieve


def test_rrf_adds_both_lists_and_keeps_a_lone_hit():
    ranked = dict(fuse([["a", "b"], ["a", "c"]]))
    assert ranked["a"] == pytest.approx(1 / 61 + 1 / 61)
    assert ranked["b"] == pytest.approx(1 / 62)
    assert ranked["c"] == pytest.approx(1 / 62)


def test_agreement_requires_both_lists_and_a_wide_gap():
    wide = fuse([["A", "B"], ["A", "C"]])
    assert agreed_parent(wide, ["A", "B"], ["A", "C"]) == "A"
    tied = fuse([["A", "B"], ["B", "A"]])
    assert agreed_parent(tied, ["A", "B"], ["B", "A"]) is None


def _chunk(chunk_id, parent_id, text):
    return {"chunk_id": chunk_id, "parent_id": parent_id, "embed_text": text}


def _parent(text):
    return {
        "text": text,
        "heading_path": "H",
        "source_path": "guide.md",
        "file_sha256": "abc123",
        "page_start": 1,
        "page_end": 2,
        "start_char": 4,
        "end_char": 4 + len(text),
    }


class Stub:
    def __init__(self, dense, bm25, parents, tau=None):
        self._dense = dense
        self._bm25 = bm25
        self._parents = parents
        self._tau = tau
        self.model_id = "test"
        self.model_revision = "rev"
        self.rerank_calls = 0
        self.doc_filter = None

    def dense_search(self, vector, k, doc_id=None):
        self.doc_filter = doc_id
        return self._dense[:k]

    def bm25_search(self, text, k, doc_id=None):
        self.doc_filter = doc_id
        return self._bm25[:k]

    def get_parents(self, ids):
        return {parent_id: self._parents[parent_id] for parent_id in ids}

    def get_tau(self, key):
        return self._tau


def _rerank(query, texts):
    scores = []
    for text in texts:
        if "alpha" in text:
            scores.append(1.0)
        elif "beta" in text:
            scores.append(0.9)
        else:
            scores.append(0.2)
    return scores


def test_fast_path_skips_the_reranker_when_both_indexes_agree():
    stub = Stub(
        [_chunk("c1", "A", "alpha sku"), _chunk("c2", "B", "other")],
        [_chunk("c1", "A", "alpha sku"), _chunk("c3", "C", "third")],
        {"A": _parent("alpha sku"), "B": _parent("other"), "C": _parent("third")},
    )

    def boom(query, texts):
        stub.rerank_calls += 1
        return _rerank(query, texts)

    result = retrieve(stub, "alpha", lambda text: [1.0, 0.0], rerank=boom)
    assert stub.rerank_calls == 0
    assert result.reason == ""
    assert len(result.hits) == 1
    assert result.hits[0].parent_id == "A"
    assert result.hits[0].confident
    assert result.hits[0].source_path == "guide.md"
    assert result.hits[0].page_start == 1
    assert result.hits[0].start_char == 4


def test_disagreement_reranks_and_abstains_under_tau():
    stub = Stub(
        [_chunk("c1", "A", "alpha"), _chunk("c2", "B", "beta")],
        [_chunk("c2", "B", "beta"), _chunk("c1", "A", "alpha")],
        {"A": _parent("alpha"), "B": _parent("beta")},
        tau=1.5,
    )
    result = retrieve(stub, "alpha", lambda text: [1.0], rerank=_rerank)
    assert result.hits == []
    assert result.reason == "no_confident_hit"


def test_tau_is_off_until_a_value_is_stored_and_the_band_drops_a_far_second():
    stub = Stub(
        [_chunk("c1", "A", "alpha"), _chunk("c2", "B", "other")],
        [_chunk("c2", "B", "other"), _chunk("c1", "A", "alpha")],
        {"A": _parent("alpha body"), "B": _parent("other body")},
    )
    result = retrieve(stub, "alpha", lambda text: [1.0], rerank=_rerank)
    assert [hit.parent_id for hit in result.hits] == ["A"]
    assert result.hits[0].confident
    assert result.reason == ""


def test_close_scores_stay_together():
    stub = Stub(
        [_chunk("c1", "A", "alpha"), _chunk("c2", "B", "beta")],
        [_chunk("c2", "B", "beta"), _chunk("c1", "A", "alpha")],
        {"A": _parent("alpha"), "B": _parent("beta")},
    )
    result = retrieve(stub, "alpha", lambda text: [1.0], rerank=_rerank)
    assert [hit.parent_id for hit in result.hits] == ["A", "B"]
    assert not result.hits[0].confident


def test_empty_query_is_refused():
    stub = Stub([], [], {})
    with pytest.raises(QueryError):
        retrieve(stub, "   ", lambda text: [1.0])


def test_doc_filter_is_passed_through():
    stub = Stub([_chunk("c1", "A", "alpha")], [_chunk("c1", "A", "alpha")], {"A": _parent("alpha")})
    retrieve(stub, "alpha", lambda text: [1.0], rerank=_rerank, doc_id="guide.md", mode="bm25")
    assert stub.doc_filter == "guide.md"
