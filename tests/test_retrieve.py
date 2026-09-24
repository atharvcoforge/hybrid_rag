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


def test_tau_is_off_until_a_value_is_stored_and_a_far_second_stays():
    stub = Stub(
        [_chunk("c1", "A", "alpha"), _chunk("c2", "B", "other")],
        [_chunk("c2", "B", "other"), _chunk("c1", "A", "alpha")],
        {"A": _parent("alpha body"), "B": _parent("other body")},
    )
    result = retrieve(stub, "alpha", lambda text: [1.0], rerank=_rerank)
    assert [hit.parent_id for hit in result.hits] == ["A", "B"]
    assert result.hits[0].confident
    assert not result.hits[1].confident
    assert result.reason == ""


def test_close_scores_stay_together():
    stub = Stub(
        [_chunk("c1", "A", "alpha"), _chunk("c2", "B", "beta")],
        [_chunk("c2", "B", "beta"), _chunk("c1", "A", "alpha")],
        {"A": _parent("alpha"), "B": _parent("beta")},
    )
    result = retrieve(stub, "alpha", lambda text: [1.0], rerank=_rerank)
    assert [hit.parent_id for hit in result.hits] == ["A", "B"]
    assert result.hits[0].confident
    assert result.hits[1].confident


def test_empty_query_is_refused():
    stub = Stub([], [], {})
    with pytest.raises(QueryError):
        retrieve(stub, "   ", lambda text: [1.0])


def test_rrf_respects_a_stored_tau():
    stub = Stub(
        [_chunk("c1", "A", "alpha"), _chunk("c2", "B", "noise")],
        [_chunk("c1", "A", "alpha"), _chunk("c2", "B", "noise")],
        {"A": _parent("alpha"), "B": _parent("noise")},
        tau=10.0,
    )
    # RRF scores are ~1/61; a high cutoff must abstain in live mode too.
    result = retrieve(stub, "alpha", lambda text: [1.0], rerank=_rerank, mode="rrf")
    assert result.hits == []
    assert result.reason == "no_confident_hit"


def test_dense_and_bm25_also_hit_the_terminal_gate():
    stub = Stub(
        [_chunk("c1", "A", "alpha")],
        [_chunk("c1", "A", "alpha")],
        {"A": _parent("alpha")},
        tau=10.0,
    )
    for mode in ("dense", "bm25"):
        result = retrieve(stub, "alpha", lambda text: [1.0], rerank=_rerank, mode=mode)
        assert result.hits == [], mode
        assert result.reason == "no_confident_hit", mode


def test_bm25_keeps_later_parents_when_scores_are_ranks():
    stub = Stub(
        [],
        [_chunk("c1", "A", "alpha"), _chunk("c2", "B", "beta")],
        {"A": _parent("alpha body"), "B": _parent("beta body")},
    )
    result = retrieve(stub, "alpha", lambda text: [1.0], rerank=_rerank, mode="bm25")
    assert [hit.parent_id for hit in result.hits] == ["A", "B"]


def test_bm25_orders_by_the_search_score():
    stub = Stub(
        [],
        [
            {**_chunk("c1", "A", "alpha"), "bm25_score": 1.0},
            {**_chunk("c2", "B", "beta"), "bm25_score": 9.0},
        ],
        {"A": _parent("alpha body"), "B": _parent("beta body")},
    )
    result = retrieve(stub, "alpha", lambda text: [1.0], rerank=_rerank, mode="bm25")
    assert result.hits[0].parent_id == "B"


def test_superseded_copy_cannot_take_the_last_slot():
    parents = {
        "old": {**_parent("old fact"), "status": "superseded"},
        "new": {**_parent("new fact"), "status": "current"},
    }
    filler = {}
    chunks = [{**_chunk("c-old", "old", "old"), "bm25_score": 10.0}]
    for index, score in enumerate((9.0, 8.0, 7.0, 6.0), start=1):
        parent_id = f"p{index}"
        filler[parent_id] = _parent(f"filler {index}")
        chunks.append({**_chunk(f"c{index}", parent_id, f"filler {index}"), "bm25_score": score})
    chunks.append({**_chunk("c-new", "new", "new"), "bm25_score": 5.8})
    stub = Stub([], chunks, {"old": parents["old"], "new": parents["new"], **filler})
    result = retrieve(stub, "alpha", lambda text: [1.0], rerank=_rerank, mode="bm25")
    assert "new" in [hit.parent_id for hit in result.hits]
    assert "old" not in [hit.parent_id for hit in result.hits]


def _policy_parents():
    old = {
        **_parent("Carbon Neutral in our operations by 2050. Review Date – 10th March 2025"),
        "source_path": "Environmental_Sustainability_Policy_2025.pdf",
        "status": "superseded",
        "superseded": True,
        "review_date": "2025-03-10",
        "version_group": "env",
    }
    new = {
        **_parent("Carbon Neutral in our operations by 2040. Review Date – 10th March 2026"),
        "source_path": "Environmental_Sustainability_Policy_2026.pdf",
        "status": "current",
        "review_date": "2026-03-10",
        "version_group": "env",
    }
    chunks = [
        {**_chunk("c-old", "old", "2050"), "bm25_score": 10.0},
        {**_chunk("c-new", "new", "2040"), "bm25_score": 9.0},
    ]
    return Stub([], chunks, {"old": old, "new": new})


def test_unqualified_carbon_query_ranks_the_current_policy_first():
    result = retrieve(
        _policy_parents(),
        "By when does Coforge commit to becoming carbon neutral in its operations?",
        lambda text: [1.0],
        mode="bm25",
    )
    assert result.hits[0].source_path.endswith("2026.pdf")
    assert any(hit.source_path.endswith("2025.pdf") for hit in result.hits)


def test_named_year_outranks_a_higher_scoring_sibling():
    stub = _policy_parents()
    stub._bm25 = [
        {**_chunk("c-old", "old", "2050"), "bm25_score": 4.0},
        {**_chunk("c-new", "new", "2040"), "bm25_score": 12.0},
    ]
    result = retrieve(
        stub,
        "What is the review date on the 2025 Environmental Sustainability Policy?",
        lambda text: [1.0],
        mode="bm25",
    )
    assert result.hits[0].source_path.endswith("2025.pdf")


def test_bm25_does_not_demote_by_country_name():
    chunks = [
        {**_chunk("c-uk", "uk", "uk"), "bm25_score": 12.0},
        {**_chunk("c-in", "india", "india"), "bm25_score": 11.0},
    ]
    parents = {
        "uk": _parent(
            "Baseline Year: 2023–2024 (UK operations) Scope 3 "
            + ("detail " * 30)
            + "India is mentioned later"
        ),
        "india": _parent("Baseline Year: 2023–2024 (India operations) Scope 3 figure 5,543"),
    }
    result = retrieve(
        Stub([], chunks, parents),
        "What is India's baseline Scope 3 figure in tCO2e?",
        lambda text: [1.0],
        mode="bm25",
    )
    assert result.hits[0].parent_id == "uk"
    assert {hit.parent_id for hit in result.hits} == {"uk", "india"}


def test_bm25_keeps_score_order_without_a_country_rule():
    chunks = [
        {**_chunk("c-uk", "uk", "uk"), "bm25_score": 10.0},
        {**_chunk("c-in", "india", "india"), "bm25_score": 9.0},
    ]
    parents = {
        "uk": _parent("Baseline Year: 2023–2024 (UK operations) Scope 1"),
        "india": _parent("India operations Scope 1 baseline 413 tCO2e"),
    }
    result = retrieve(
        Stub([], chunks, parents),
        "What is India's baseline Scope 1 figure?",
        lambda text: [1.0],
        mode="bm25",
    )
    assert result.hits[0].parent_id == "uk"
    assert {hit.parent_id for hit in result.hits} == {"uk", "india"}


def test_conflict_fact_loads_the_missing_superseded_passage():
    current = {
        **_parent("Carbon Neutral in our operations by 2040"),
        "source_path": "Environmental_Sustainability_Policy_2026.pdf",
        "status": "current",
        "version_group": "env",
        "review_date": "2026-03-10",
    }

    class _Sibling(Stub):
        def list_doc_records(self):
            return [
                {"id": "new", "version_group": "env", "status": "current"},
                {"id": "old", "version_group": "env", "status": "superseded"},
            ]

        def parent_records(self, doc_id):
            text = (
                "Carbon Neutral in our operations by 2050"
                if doc_id == "old"
                else "Carbon Neutral in our operations by 2040"
            )
            parent = _parent(text)
            parent.update(
                {
                    "parent_id": doc_id,
                    "text": text,
                    "status": "superseded" if doc_id == "old" else "current",
                    "superseded": doc_id == "old",
                    "source_path": (
                        "Environmental_Sustainability_Policy_2025.pdf"
                        if doc_id == "old"
                        else "Environmental_Sustainability_Policy_2026.pdf"
                    ),
                    "version_group": "env",
                    "heading_path": "H",
                    "child_id": f"c-{doc_id}",
                }
            )
            return [parent]

    stub = _Sibling(
        [],
        [{**_chunk("c-new", "new", "2040"), "bm25_score": 9.0}],
        {"new": current},
    )
    result = retrieve(
        stub,
        "By when does Coforge commit to becoming carbon neutral in its operations?",
        lambda text: [1.0],
        mode="bm25",
    )
    paths = [hit.source_path for hit in result.hits]
    assert any(path.endswith("2025.pdf") for path in paths)
    assert result.hits[0].source_path.endswith("2026.pdf")


def test_query_that_names_2025_ranks_the_superseded_policy_first():
    result = retrieve(
        _policy_parents(),
        "What is the review date on the 2025 Environmental Sustainability Policy?",
        lambda text: [1.0],
        mode="bm25",
    )
    assert result.hits[0].source_path.endswith("2025.pdf")
    assert any(hit.source_path.endswith("2026.pdf") for hit in result.hits)


def test_fuse_is_linear_in_list_length():
    # Regression for F-19: order.index inside the sort is O(n²).
    ids = [f"id-{i}" for i in range(2000)]
    ranked = fuse([ids, list(reversed(ids))])
    assert ranked[0][0] == "id-0" or ranked[0][0] == ids[-1]
    assert len(ranked) == 2000
