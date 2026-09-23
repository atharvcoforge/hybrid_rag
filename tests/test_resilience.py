"""Resilience: cache, queue, circuit breaker."""

import time

import pytest

from rag.cache import AnswerCache
from rag.health import CircuitBreaker, HealthState
from rag.queue import BusyError, InferenceQueue


def test_answer_cache_is_bounded_lru():
    cache = AnswerCache(maxsize=2, ttl_s=60)
    cache[("a", 1, "cascade")] = {"answer": "1"}
    cache[("b", 1, "cascade")] = {"answer": "2"}
    cache[("c", 1, "cascade")] = {"answer": "3"}
    assert cache.get(("a", 1, "cascade")) is None
    assert cache.get(("b", 1, "cascade"))["answer"] == "2"
    assert cache.get(("c", 1, "cascade"))["answer"] == "3"


def test_answer_cache_expires():
    cache = AnswerCache(maxsize=8, ttl_s=0.01)
    key = AnswerCache.make_key("Who signed?", 1, "rrf")
    cache[key] = {"answer": "old"}
    time.sleep(0.02)
    assert cache.get(key) is None


def test_answer_cache_keys_include_generation():
    assert AnswerCache.make_key("Q", 1, "rrf") != AnswerCache.make_key("Q", 2, "rrf")


def test_answer_cache_keys_include_doc_filter():
    bare = AnswerCache.make_key("Q", 1, "rrf")
    filtered = AnswerCache.make_key("Q", 1, "rrf", "Carbon_Reduction_Plan.pdf")
    assert bare != filtered


def test_inference_queue_rejects_when_full():
    q = InferenceQueue(maxsize=1)
    q.acquire()
    with pytest.raises(BusyError):
        q.acquire()
    q.release()
    q.acquire()
    q.release()


def test_circuit_opens_after_threshold_failures():
    breaker = CircuitBreaker(fail_threshold=2, reset_s=60)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed"


def test_health_snapshot_lists_degradation():
    state = HealthState(dense_ok=False)
    state.note("Semantic search unavailable — keyword results only")
    snap = state.snapshot()
    assert snap["degraded"] is True
    assert "Semantic search" in snap["messages"][0]
