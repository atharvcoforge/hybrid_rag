"""Cover server routes and the query iterator branches."""

import asyncio
import json
import runpy
from pathlib import Path

import pytest
from starlette.requests import Request

from rag.health import CircuitBreaker, HealthState
from rag.models import (
    EMBED_MODEL,
    EMBED_REVISION,
    PIPELINE_VERSION,
    Hit,
    Ingested,
    Retrieval,
)
from rag.queue import InferenceQueue
from rag.server import (
    IngestBody,
    QueryBody,
    _verdict,
    ask,
    build_report,
    docs_list,
    eval_get,
    eval_post,
    health,
    index_dir,
    ingest_route,
    iter_query,
    live_mode,
    ready,
    span_rate,
)
from rag.telemetry import StageBudgetExceeded


@pytest.fixture(autouse=True)
def _restore_server_globals():
    from rag import server

    health_before = server._health
    circuit_before = server._circuit
    answers_before = server._answers
    queue_before = server._queue
    yield
    server._health = health_before
    server._circuit = circuit_before
    server._answers = answers_before
    server._queue = queue_before
    server._health.messages.clear()
    server._health.writer_ok = True
    server._health.warmed = False
    server._circuit.failures = 0
    server._circuit.opened_at = None
    server._answers.clear()


def _hit(text="14,644 tCO2e in the plan [1]"):
    return Hit(
        parent_id="p",
        parent_text=text,
        heading_path="H",
        source_path="doc.pdf",
        file_sha256="abc",
        page_start=1,
        page_end=2,
        start_char=0,
        end_char=4,
        child_id="c",
        score=0.4,
        confident=True,
    )


def _events(text, search, write, **kwargs):
    return list(iter_query(text, search, write, kwargs.pop("mode", "rrf"), **kwargs))


def test_empty_query_and_defaults():
    events = _events("   ", lambda *_: Retrieval(hits=[]), lambda *_: iter(()))
    assert events[1][0] == "error"


def test_search_budget_and_warnings():
    circuit = CircuitBreaker(fail_threshold=1)
    state = HealthState(writer_ok=True)

    def search(*_args):
        raise StageBudgetExceeded("dense", 10, 20)

    events = _events("hello", search, lambda *_: iter(()), circuit=circuit, health=state, cache={})
    assert events[-1][0] == "error"
    assert circuit.failures == 1

    def warned(*_args):
        return Retrieval(hits=[], reason="no_confident_hit", warnings=["slow"])

    events = _events("hello", warned, lambda *_: iter(()), health=state, cache={})
    meta = next(data for name, data in events if name == "meta")
    assert "slow" in meta["degraded"]


def test_extractive_when_writer_is_down():
    state = HealthState(writer_ok=False)

    def search(*_args):
        return Retrieval(hits=[_hit()])

    events = _events("hello", search, lambda *_: iter(()), health=state, cache={})
    assert any(data.get("extractive") for _name, data in events if isinstance(data, dict))


def test_writer_failures_and_empty_pieces():
    state = HealthState(writer_ok=True)
    circuit = CircuitBreaker()

    def search(*_args):
        return Retrieval(hits=[_hit()])

    def empty_then_raise(_text, _hits):
        yield ""
        raise StageBudgetExceeded("generate_complete", 10, 50)

    events = _events("hello", search, empty_then_raise, health=state, circuit=circuit, cache={})
    assert events[-1][1]["message"] == "writer unavailable"

    def partial(_text, _hits):
        yield "partial "
        raise RuntimeError("stall")

    events = _events("hello again", search, partial, health=state, circuit=circuit, cache={})
    assert events[-1][1]["partial"] == "partial "
    assert events[-1][1]["message"] == "Answer incomplete"


def test_span_rate_and_verdict_edges():
    rows = [{"q": "q", "expect": "14,644", "must_contain": "14,644", "doc_id": "doc.pdf"}]

    def search(_q, _mode):
        return Retrieval(hits=[_hit()], reason="no_confident_hit")

    assert span_rate([], search, lambda *_: "", "rrf") == (None, 0)
    rate, n = span_rate(rows, search, lambda *_: "14,644", "rrf")
    assert rate == 0.0 and n == 1

    def good(_q, _mode):
        return Retrieval(hits=[_hit()])

    rate, n = span_rate(rows, good, lambda *_: "14,644", "rrf")
    assert rate == 1.0

    text = _verdict(
        {
            "live_mode": "rrf",
            "scores": [
                {"kind": "all", "mode": "rrf", "recall": 1, "mrr": 1, "abstain": 0, "p50": 1, "p95": 2},
                {"kind": "all", "mode": "rerank", "recall": 0.5, "mrr": 0.5, "abstain": 0, "p50": 9},
            ],
            "span_hit": 0.2,
            "span_n": 4,
        }
    )
    assert "weak" in text
    strong = _verdict(
        {
            "live_mode": "rrf",
            "scores": [{"kind": "all", "mode": "rrf", "recall": 1, "mrr": 1, "abstain": 0, "p50": 1}],
            "span_hit": 0.9,
            "span_n": 4,
        }
    )
    assert "stays" in strong


def test_lifespan_and_routes(monkeypatch, tmp_path):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("CORPUS", str(tmp_path / "docs"))
    monkeypatch.setenv("EVAL_OUT", str(tmp_path / "latest.json"))
    monkeypatch.setenv("GOLDEN", str(tmp_path / "rows.jsonl"))
    monkeypatch.setenv("LIVE_MODE", "rrf")
    monkeypatch.setattr("rag.server.writer_up", lambda: False)
    monkeypatch.setattr("rag.embed.load_embedder", lambda: (_ for _ in ()).throw(RuntimeError("no weights")))

    from rag.server import app, lifespan

    async def open_ok():
        async with lifespan(app):
            assert health()["index"] in {True, False}

    asyncio.run(open_ok())
    assert ready()["ready"] is True
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# T\n\nhello\n", encoding="utf-8")
    monkeypatch.setattr(
        "rag.server.ingest",
        lambda *_a, **_k: [Ingested("a.md", "indexed", 1, warnings=["flagged"])],
    )
    body = ingest_route(None, IngestBody())
    assert body["items"][0]["doc_id"] == "a.md"
    listed = docs_list()
    assert "documents" in listed
    monkeypatch.setattr("rag.server.ingest", lambda *_a, **_k: [])

    class Closed:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            raise RuntimeError("closed")

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr("rag.store.Index", Closed)
    ingest_route(None, None)

    monkeypatch.delenv("LIVE_MODE", raising=False)
    assert live_mode() == "cascade"
    path = tmp_path / "latest.json"
    path.write_text(json.dumps({"live_mode": "bm25", "scores": []}), encoding="utf-8")
    monkeypatch.setenv("EVAL_OUT", str(path))
    assert live_mode() == "bm25"
    assert eval_get()["live_mode"] == "bm25"
    path.unlink()
    assert eval_get()["scores"] == []

    class Boom:
        def __init__(self, *_a, **_k):
            pass

        def open(self):
            raise RuntimeError("locked")

    monkeypatch.setattr("rag.store.Index", Boom)

    async def open_bad():
        async with lifespan(app):
            return

    asyncio.run(open_bad())


def test_query_route_and_eval_post(monkeypatch, tmp_path):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("EVAL_OUT", str(tmp_path / "out.json"))
    monkeypatch.setenv("API_TOKEN", "secret")
    monkeypatch.setattr(
        "rag.server._search",
        lambda *_a, **_k: Retrieval(hits=[], reason="no_confident_hit"),
    )
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/query",
        "raw_path": b"/api/query",
        "query_string": b"",
        "headers": [(b"x-request-id", b"trace-1")],
        "client": ("127.0.0.1", 123),
        "server": ("127.0.0.1", 8000),
    }
    response = ask(QueryBody(q="hello"), Request(scope))

    async def drain():
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(drain())
    assert chunks
    from rag import server

    original_queue = server._queue
    try:
        server._queue = InferenceQueue(maxsize=1)
        server._queue.acquire()
        with pytest.raises(Exception) as exc:
            ask(QueryBody(q="busy"), Request(scope))
        assert getattr(exc.value, "status_code", None) == 429
        server._queue.release()
        monkeypatch.setattr(
            "rag.server.build_report",
            lambda *_a, **_k: {"scores": [], "live_mode": "rrf", "verdict": "ok"},
        )
        report = eval_post(None)
        assert report["live_mode"] == "rrf"
        assert Path(tmp_path / "out.json").exists()
        server._queue = InferenceQueue(maxsize=1)
        server._queue.acquire()
        with pytest.raises(Exception) as busy:
            eval_post(None)
        assert getattr(busy.value, "status_code", None) == 429
    finally:
        server._queue = original_queue


def test_build_report_without_a_writer(monkeypatch, tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        json.dumps({"id": "1", "q": "q", "kind": "lexical", "must_contain": "x", "doc_id": "d"}) + "\n",
        encoding="utf-8",
    )
    from rag.evaluate import Score

    monkeypatch.setattr("rag.server.writer_up", lambda: False)
    monkeypatch.setattr(
        "rag.server.evaluate",
        lambda *_a, **_k: ([Score("rrf", "all", 1.0, 1.0, 0.0, 1)], {"rrf": 0.1}),
    )
    monkeypatch.setattr("rag.server.pick_live", lambda lines: "rrf")
    report = build_report(tmp_path / "index", golden, lambda *_: Retrieval(hits=[]), None)
    assert report["live_mode"] == "rrf"
    assert report["span_hit"] is None
    monkeypatch.setattr("rag.server.writer_up", lambda: True)

    def fake_eval(index, rows, ask, split=None):
        del index, split
        for row in rows:
            ask("rrf", row)
        return [Score("rrf", "all", 1.0, 1.0, 0.0, 1)], 0.1

    monkeypatch.setattr("rag.server.evaluate", fake_eval)
    monkeypatch.setattr("rag.retrieve.retrieve", lambda *_a, **_k: Retrieval(hits=[]))
    report = build_report(tmp_path / "index2", golden, lambda *_: Retrieval(hits=[]), lambda *_: "ans")
    assert "span_hit" in report


def test_server_main_guard(monkeypatch):
    monkeypatch.setattr("uvicorn.run", lambda *_a, **_k: None)
    runpy.run_module("rag.server", run_name="__main__")


def test_remaining_query_branches(monkeypatch):
    from rag.gates import GateResult
    from rag.server import _health, _search, ready

    state = HealthState(writer_ok=True)
    cache = {}

    def search(*_args):
        return Retrieval(hits=[_hit()])

    def blanks(_text, _hits):
        yield ""
        yield "14,644 [1]"
        yield " more"

    events = _events("cited fact", search, blanks, health=state, cache=cache)
    assert any(name == "token" for name, _data in events)

    def nothing(_text, _hits):
        if False:
            yield ""

    _events("no tokens", search, nothing, health=state, cache={})
    empty_cache: dict = {}

    def abstain(*_args):
        return Retrieval(hits=[], reason="no_confident_hit")

    _events("absent fact", abstain, lambda *_: iter(()), health=state, cache=empty_cache)
    again = _events("absent fact", abstain, lambda *_: iter(()), health=state, cache=empty_cache)
    assert any(data.get("cached") and data.get("answer") == "" for _name, data in again if isinstance(data, dict))

    saved_key_events = _events("cited fact", search, blanks, health=state, cache=cache)
    assert any(data.get("cached") for _name, data in saved_key_events if isinstance(data, dict))

    monkeypatch.setattr(
        "rag.server.check_all",
        lambda *_a, **_k: GateResult(True, "", state="verified", conflict="2040 vs 2050", answer="2040 [1]"),
    )
    conflict = _events("carbon", search, lambda *_: iter(["2040 [1]"]), health=state, cache={})
    assert any(data.get("conflict") for _name, data in conflict if isinstance(data, dict))

    def miss(_q, _mode):
        return Retrieval(hits=[_hit(text="other")])

    rate, _n = span_rate(
        [{"q": "q", "expect": "14,644", "must_contain": "14,644", "doc_id": "missing.pdf"}],
        miss,
        lambda *_: "nope",
        "rrf",
    )
    assert rate == 0.0
    rate, _n = span_rate(
        [{"q": "q", "expect": "14,644", "must_contain": "14,644", "doc_id": "doc.pdf"}],
        search,
        lambda *_: "nope",
        "rrf",
    )
    assert rate == 0.0

    def boom(*_a, **_k):
        raise RuntimeError("index down")

    monkeypatch.setattr("rag.server._search", boom)
    monkeypatch.setattr("rag.server.await_disconnected", lambda _request: True)
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/query",
        "raw_path": b"/api/query",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8000),
    }
    response = ask(QueryBody(q="hello"), Request(scope))

    async def drain():
        return [chunk async for chunk in response.body_iterator]

    asyncio.run(drain())
    import time

    time.sleep(0.05)
    calls = {"n": 0}

    def flaky(_request):
        calls["n"] += 1
        return calls["n"] > 2

    monkeypatch.setattr("rag.server.await_disconnected", flaky)
    monkeypatch.setattr("rag.server._search", lambda *_a, **_k: Retrieval(hits=[], reason="no_confident_hit"))
    slow = ask(QueryBody(q="slow"), Request(scope))

    async def drain_slow():
        return [chunk async for chunk in slow.body_iterator]

    asyncio.run(drain_slow())
    monkeypatch.setattr(
        "rag.server.check_all",
        lambda *_a, **_k: GateResult(True, "", state="verified", answer=""),
    )
    plain = _events("plain success", search, lambda *_: iter(()), health=state, cache={})
    assert any(data.get("answer") == "" for _name, data in plain if isinstance(data, dict))

    def slow_search(*_a, **_k):
        time.sleep(0.2)
        return Retrieval(hits=[], reason="no_confident_hit")

    monkeypatch.setattr("rag.server._search", slow_search)
    ticks = {"n": 0}

    def disconnect_on_wait(_request):
        ticks["n"] += 1
        return ticks["n"] >= 3

    monkeypatch.setattr("rag.server.await_disconnected", disconnect_on_wait)
    waiting = ask(QueryBody(q="wait"), Request(scope))

    async def drain_wait():
        return [chunk async for chunk in waiting.body_iterator]

    asyncio.run(drain_wait())
    monkeypatch.setattr("rag.server.query", lambda *_a, **_k: Retrieval(hits=[]))
    _search("hello", "rrf")
    _health.warmed = False
    with pytest.raises(Exception) as exc:
        ready()
    assert getattr(exc.value, "status_code", None) == 503
    _health.warmed = True


def test_lifespan_writer_up_and_close_error(monkeypatch, tmp_path):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setattr("rag.server.writer_up", lambda: True)
    monkeypatch.setattr("rag.embed.load_embedder", lambda: object())

    class Store:
        def __init__(self, *_a, **_k):
            pass

        def open(self):
            return None

        def index_generation(self):
            return 3

        def close(self):
            raise RuntimeError("close")

    monkeypatch.setattr("rag.store.Index", Store)
    from rag.server import app, lifespan

    async def go():
        async with lifespan(app):
            return

    asyncio.run(go())
    monkeypatch.setenv("LIVE_MODE", "bm25")
    assert live_mode() == "bm25"


def test_env_helpers(monkeypatch, tmp_path):
    monkeypatch.delenv("INDEX_DIR", raising=False)
    monkeypatch.delenv("CORPUS", raising=False)
    assert index_dir() == Path("index")
    from rag.server import corpus_root, eval_path, golden_path

    assert corpus_root() == Path("documents")
    monkeypatch.setenv("EVAL_OUT", str(tmp_path / "e.json"))
    monkeypatch.setenv("GOLDEN", str(tmp_path / "g.jsonl"))
    assert eval_path() == tmp_path / "e.json"
    assert golden_path() == tmp_path / "g.jsonl"
    assert EMBED_MODEL and EMBED_REVISION and PIPELINE_VERSION
