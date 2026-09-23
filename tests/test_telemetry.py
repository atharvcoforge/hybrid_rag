import json
import os
import threading
import time
import uuid

import pytest

from rag.telemetry import (
    StageBudgetExceeded,
    StageTimer,
    bind_trace,
    content_logging_enabled,
    log_event,
    mint_trace_id,
    new_request_id,
    percentile,
    span,
    uuid7,
)


def test_percentile_empty_is_zero():
    assert percentile([], 50) == 0.0


def test_percentile_p50_p95_p99_on_known_series():
    samples = [float(n) for n in range(1, 101)]
    assert percentile(samples, 50) == 50.0
    assert percentile(samples, 95) == 95.0
    assert percentile(samples, 99) == 99.0


def test_percentile_single_sample():
    assert percentile([7.5], 99) == 7.5


def test_stage_timer_records_named_stages():
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    timer = StageTimer(clock=now)
    with timer.measure("embed"):
        clock["t"] += 0.01
    with timer.measure("dense"):
        clock["t"] += 0.02
    ms = timer.as_ms()
    assert abs(ms["embed"] - 10.0) < 0.5
    assert abs(ms["dense"] - 20.0) < 0.5
    assert set(ms) == {"embed", "dense"}


def test_request_id_is_short_and_unique():
    a = new_request_id()
    b = new_request_id()
    assert a != b
    assert 8 <= len(a) <= 16
    assert a.isalnum()


def test_uuid7_is_version_7():
    value = uuid7()
    assert isinstance(value, uuid.UUID)
    assert value.version == 7


def test_mint_trace_id_accepts_inbound():
    assert mint_trace_id("client-trace-1") == "client-trace-1"
    minted = mint_trace_id(None)
    assert uuid.UUID(minted).version == 7


def test_log_event_is_one_json_line(capsys):
    log_event("retrieve", request_id="abc123", mode="rrf", dense_ms=12.5)
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["event"] == "retrieve"
    assert payload["request_id"] == "abc123"
    assert payload["mode"] == "rrf"
    assert payload["dense_ms"] == 12.5
    assert "ts" in payload


def test_span_emits_start_and_finish(capsys):
    with bind_trace("trace-span-1"):
        with span("dense", candidates_in=3) as handle:
            handle["candidates_out"] = 2
            handle["top_score"] = 0.5
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    events = [(row["stage"], row["event"]) for row in lines]
    assert ("dense", "start") in events
    assert ("dense", "finish") in events
    finish = next(row for row in lines if row["event"] == "finish")
    assert finish["trace_id"] == "trace-span-1"
    assert finish["candidates_out"] == 2
    assert finish["top_score"] == 0.5
    assert "duration_ms" in finish


def test_span_error_records_exception_type(capsys):
    with bind_trace("trace-err"):
        with pytest.raises(RuntimeError, match="boom"):
            with span("rerank"):
                raise RuntimeError("boom")
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    error = next(row for row in lines if row["event"] == "error")
    assert error["error_type"] == "RuntimeError"
    assert error["error"] == "boom"
    assert error["stage"] == "rerank"


def test_stage_slow_fires_during_hang(capsys, monkeypatch):
    monkeypatch.setenv("STAGE_SOFT_MS_dense", "50")
    monkeypatch.setenv("STAGE_HARD_MS_dense", "5000")
    slow_at = {"t": None}

    def capture():
        # Poll stdout via capsys is awkward mid-flight; watch for log_span side effect
        # by reading after a barrier. Instead, set a flag from a custom sink — use
        # threading: the stage_slow line must appear before span exits.
        pass

    started = time.perf_counter()
    saw_slow = threading.Event()

    # Monkeypatch log_span to notice stage_slow while sleep is in progress.
    import rag.telemetry as telemetry

    original = telemetry.log_span

    def wrapped(stage, event, **kwargs):
        if event == "stage_slow":
            slow_at["t"] = time.perf_counter()
            saw_slow.set()
        return original(stage, event, **kwargs)

    monkeypatch.setattr(telemetry, "log_span", wrapped)
    with bind_trace("trace-slow"):
        with span("dense"):
            assert saw_slow.wait(2.0), "stage_slow did not fire during hang"
            assert slow_at["t"] is not None
            assert slow_at["t"] - started < 1.5  # fired well before our sleep ends
            time.sleep(0.15)  # finish after the soft warning
    assert saw_slow.is_set()


def test_hard_budget_raises(monkeypatch):
    monkeypatch.setenv("STAGE_SOFT_MS_fuse", "10")
    monkeypatch.setenv("STAGE_HARD_MS_fuse", "40")
    with bind_trace("trace-hard"):
        with pytest.raises(StageBudgetExceeded) as raised:
            with span("fuse"):
                time.sleep(0.12)
    assert raised.value.stage == "fuse"


def test_query_text_not_logged_by_default(capsys, monkeypatch):
    monkeypatch.delenv("LOG_CONTENT", raising=False)
    assert content_logging_enabled() is False
    with bind_trace("trace-content"):
        log_event("query_start", query="SECRET_PASSAGE_TEXT_XYZ", mode="rrf")
    out = capsys.readouterr().out
    assert "SECRET_PASSAGE_TEXT_XYZ" not in out
    payload = json.loads(out.strip().splitlines()[-1])
    assert "query" not in payload


def test_query_text_logged_when_enabled(capsys, monkeypatch):
    monkeypatch.setenv("LOG_CONTENT", "1")
    # force re-read via content_logging_enabled which checks env each time
    with bind_trace("trace-content-on"):
        log_event("query_start", query="SECRET_PASSAGE_TEXT_XYZ", mode="rrf")
    out = capsys.readouterr().out
    assert "SECRET_PASSAGE_TEXT_XYZ" in out
