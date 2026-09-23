"""Branch coverage for rag.telemetry."""

import uuid

import pytest

from rag.telemetry import (
    StageBudgetExceeded,
    StageTimer,
    _publish_stage,
    clear_stage_bus,
    configure_logging,
    set_stage_bus,
    span,
    uuid7,
)


def test_per_module_log_levels_and_stage_bus(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL_PARSE", "DEBUG")
    monkeypatch.setenv("LOG_LEVEL_RAG_EMBED", "WARNING")
    configure_logging("json")

    class Bus:
        def __init__(self):
            self.items = []

        def put(self, payload):
            if payload.get("event") == "error":
                raise RuntimeError("full")
            self.items.append(payload)

    bus = Bus()
    set_stage_bus(bus)
    try:
        _publish_stage({"event": "note"})
        _publish_stage({"event": "start", "stage": "parse"})
        _publish_stage({"event": "error", "stage": "parse"})
        assert [item["event"] for item in bus.items] == ["start"]
    finally:
        clear_stage_bus()


def test_uuid7_prefers_stdlib_and_timer_default(monkeypatch):
    sentinel = uuid.UUID(int=7)
    monkeypatch.setattr(uuid, "uuid7", lambda: sentinel, raising=False)
    assert uuid7() == sentinel
    assert StageTimer().get("missing", 1.5) == 1.5


def test_span_hard_budget_without_watcher(monkeypatch):
    monkeypatch.setenv("STAGE_SOFT_MS_covstage", "60000")
    monkeypatch.setenv("STAGE_HARD_MS_covstage", "-1")
    with pytest.raises(StageBudgetExceeded), span("covstage"):
        pass
