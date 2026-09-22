import json
import time

from rag.telemetry import StageTimer, log_event, new_request_id, percentile


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


def test_log_event_is_one_json_line(capsys):
    log_event("retrieve", request_id="abc123", mode="rrf", dense_ms=12.5)
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["event"] == "retrieve"
    assert payload["request_id"] == "abc123"
    assert payload["mode"] == "rrf"
    assert payload["dense_ms"] == 12.5
    assert "ts" in payload
