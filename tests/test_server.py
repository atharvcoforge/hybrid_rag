from rag.models import Hit, Retrieval
from rag.server import iter_query


def test_abstain_does_not_call_the_writer():
    def search(_text, _mode):
        return Retrieval(hits=[], reason="no_confident_hit")

    def write(_text, _hits):
        raise AssertionError("writer called")

    events = list(iter_query("what is the salary", search, write, "cascade", cache={}))
    assert events[0][0] == "meta"
    assert events[0][1]["reason"] == "no_confident_hit"
    assert events[0][1]["request_id"]
    assert events[-1][0] == "done"
    assert events[-1][1]["first_token_ms"] is None
    assert events[-1][1]["answer"] == ""
    assert events[-1][1]["request_id"] == events[0][1]["request_id"]


def _hit():
    return Hit(
        parent_id="p",
        parent_text="Signed by John Speight",
        heading_path="",
        source_path="Carbon_New_2040.pdf",
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
    first = list(iter_query("Who signed?", search, write, "rrf", cache))
    second = list(iter_query("  who   signed? ", search, write, "rrf", cache))
    third = list(iter_query("When is net zero?", search, write, "rrf", cache))
    assert calls == {"search": 2, "write": 2}
    assert seen[0] == "Who signed?"
    assert second[0][1]["cached"] is True
    assert second[1] == ("token", {"t": "John Speight signed the plan [1]."})
    assert first[-1][1]["answer"] == "John Speight signed the plan [1]."
    hit = first[0][1]["hits"][0]
    assert hit["cite"] == 1
    assert hit["title"] == "Carbon Reduction Plan"
    assert hit["source"] == "Carbon_New_2040.pdf"
    assert second[0][1]["hits"][0]["cite"] == 1
    assert third[0][1]["cached"] is False


def test_gate_withholds_uncited_answer():
    def search(_text, _mode):
        return Retrieval(hits=[_hit()])

    def write(_text, _hits):
        yield "John Speight"

    events = list(iter_query("Who signed?", search, write, "rrf", cache={}))
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

    events = list(
        iter_query(
            "Who signed?",
            search,
            write,
            "rrf",
            cache={},
            circuit=breaker,
            health=health,
        )
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
    list(iter_query("Who signed?", search, write, "rrf", cache=cache, generation=1))
    list(iter_query("Who signed?", search, write, "rrf", cache=cache, generation=1))
    assert calls["search"] == 1
    list(iter_query("Who signed?", search, write, "rrf", cache=cache, generation=2))
    assert calls["search"] == 2
