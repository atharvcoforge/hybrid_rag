from rag.models import Hit, Retrieval
from rag.server import iter_query


def _events(text, search, write, mode="cascade", **kwargs):
    return list(iter_query(text, search, write, mode, **kwargs))


def _meta(events):
    return next(data for name, data in events if name == "meta")


def test_abstain_does_not_call_the_writer():
    def search(_text, _mode):
        return Retrieval(hits=[], reason="no_confident_hit")

    def write(_text, _hits):
        raise AssertionError("writer called")

    events = _events("what is the salary", search, write, "cascade", cache={})
    assert events[0][0] == "trace"
    assert events[0][1]["trace_id"]
    meta = _meta(events)
    assert meta["reason"] == "no_confident_hit"
    assert meta["request_id"]
    assert events[-1][0] == "done"
    assert events[-1][1]["first_token_ms"] is None
    assert events[-1][1]["answer"] == ""
    assert events[-1][1]["request_id"] == meta["request_id"]
    assert events[-1][1]["trace_id"] == events[0][1]["trace_id"]


def _hit():
    return Hit(
        parent_id="p",
        parent_text="Signed by John Speight",
        heading_path="",
        source_path="Carbon_Reduction_Plan.pdf",
        file_sha256="abc",
        page_start=3,
        page_end=3,
        start_char=0,
        end_char=10,
        child_id="c",
        score=0.2,
        confident=False,
    )


def test_verdict_survives_a_partial_report():
    from rag.server import _verdict

    text = _verdict({"live_mode": "rrf", "scores": [], "span_hit": None, "span_n": 0})
    assert "no overall score row" in text
    assert "Rerank scores are missing" in text
    assert "Still not a production service" in text


def test_repeat_skips_retrieve_and_writer():
    calls = {"search": 0, "write": 0}
    seen = []

    def search(text, _mode):
        calls["search"] += 1
        seen.append(text)
        return Retrieval(hits=[_hit()])

    def write(_text, _hits):
        calls["write"] += 1
        yield "John Speight signed the plan [1]."

    cache = {}
    first = _events("Who signed?", search, write, "rrf", cache=cache)
    second = _events("  who   signed? ", search, write, "rrf", cache=cache)
    third = _events("When is net zero?", search, write, "rrf", cache=cache)
    assert calls == {"search": 2, "write": 2}
    assert seen[0] == "Who signed?"
    assert second[1][1]["cached"] is True  # after trace
    assert ("token", {"t": "John Speight signed the plan [1]."}) in second
    assert first[-1][1]["answer"] == "John Speight signed the plan [1]."
    hit = _meta(first)["hits"][0]
    assert hit["cite"] == 1
    # Titles come from the index; fake hits without title fall back to filename.
    assert hit["title"] == "Carbon_Reduction_Plan.pdf"
    assert hit["source"] == "Carbon_Reduction_Plan.pdf"
    assert _meta(second)["hits"][0]["cite"] == 1
    assert _meta(third)["cached"] is False


def test_gate_withholds_uncited_answer():
    def search(_text, _mode):
        return Retrieval(hits=[_hit()])

    def write(_text, _hits):
        yield "John Speight"

    events = _events("Who signed?", search, write, "rrf", cache={})
    assert events[-1][1]["answer"] == "The documents do not say."
    assert events[-1][1]["verification"]["reason"] == "missing_citation"


def test_extractive_when_circuit_is_open():
    from rag.health import CircuitBreaker, HealthState

    breaker = CircuitBreaker(fail_threshold=1, reset_s=60)
    breaker.record_failure()
    health = HealthState(writer_ok=True)

    def search(_text, _mode):
        return Retrieval(hits=[_hit()])

    def write(_text, _hits):
        raise AssertionError("writer must not run")

    events = _events(
        "Who signed?",
        search,
        write,
        "rrf",
        cache={},
        circuit=breaker,
        health=health,
    )
    assert events[-1][1].get("extractive") is True
    assert events[-1][1]["answer"] == ""


def test_cache_misses_after_index_generation_bump():
    calls = {"search": 0}

    def search(_text, _mode):
        calls["search"] += 1
        return Retrieval(hits=[_hit()])

    def write(_text, _hits):
        yield "John Speight signed the plan [1]."

    cache = {}
    _events("Who signed?", search, write, "rrf", cache=cache, generation=1)
    _events("Who signed?", search, write, "rrf", cache=cache, generation=1)
    assert calls["search"] == 1
    _events("Who signed?", search, write, "rrf", cache=cache, generation=2)
    assert calls["search"] == 2


def test_inbound_trace_id_is_honoured():
    def search(_text, _mode):
        return Retrieval(hits=[], reason="no_confident_hit")

    def write(_text, _hits):
        raise AssertionError("writer called")

    events = _events(
        "x",
        search,
        write,
        cache={},
        trace_id="inbound-trace-abc",
    )
    assert events[0] == ("trace", {"trace_id": "inbound-trace-abc", "request_id": "inbound-trace-abc"})


def test_carbon_neutrality_conflict_discloses_2040_and_2050():
    """Part D target: 2040 from current + name superseded 2050 — never silent pick."""

    def search(_text, _mode):
        stale = Hit(
            parent_id="stale",
            parent_text='We, at Coforge, are committed to become "Carbon Neutral in our operations by 2050".',
            heading_path="",
            source_path="Environmental_Sustainability_Policy_2025.pdf",
            file_sha256="a",
            page_start=4,
            page_end=4,
            start_char=0,
            end_char=80,
            child_id="c1",
            score=0.9,
            confident=False,
            superseded=True,
            status="superseded",
            version_group="env",
            superseded_by="Environmental_Sustainability_Policy_2026.pdf",
            title="Environmental Sustainability Policy",
        )
        current = Hit(
            parent_id="cur",
            parent_text='We, at Coforge, are committed to become "Carbon Neutral in our operations by 2040".',
            heading_path="",
            source_path="Environmental_Sustainability_Policy_2026.pdf",
            file_sha256="b",
            page_start=4,
            page_end=4,
            start_char=0,
            end_char=80,
            child_id="c2",
            score=0.9,
            confident=False,
            superseded=False,
            status="current",
            version_group="env",
            title="Environmental Sustainability Policy",
        )
        return Retrieval(hits=[stale, current])

    def write(_text, _hits):
        yield "The documents do not say."

    events = _events(
        "By when does Coforge commit to becoming carbon neutral in its operations?",
        search,
        write,
        "rrf",
        cache={},
    )
    answer = events[-1][1]["answer"]
    assert "2040" in answer
    assert "2050" in answer
    assert "Environmental_Sustainability_Policy_2025.pdf" in answer
    assert "superseded" in answer.lower()
    assert "do not say" not in answer.lower()
    done = events[-1][1]
    assert done.get("conflict") or (done.get("verification") or {}).get("conflict")
    hits = _meta(events)["hits"]
    assert any(h.get("superseded") for h in hits)
