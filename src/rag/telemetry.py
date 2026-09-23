"""Per-request traces, stage spans, and structured logs.

Every request carries a ``trace_id`` in a ContextVar. Every stage emits
``start`` then ``finish`` or ``error`` so an open span is visible while a
stage is stuck. Soft budgets fire ``stage_slow`` *during* the hang; hard
budgets raise ``StageBudgetExceeded``.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_trace_local = threading.local()
_stage_bus_local = threading.local()
_log_format = os.environ.get("LOG_FORMAT", "json").strip().lower()

# Soft = warn while still running. Hard = raise StageBudgetExceeded.
# Tuned against suite.yaml SLOs (retrieve p95 400 ms, ttft p95 1200 ms).
DEFAULT_SOFT_MS: dict[str, float] = {
    "parse": 5_000,
    "normalize": 2_000,
    "chunk": 5_000,
    "contextualize": 30_000,
    "embed": 30_000,
    "embed_query": 400,
    "dense": 100,
    "lexical": 100,
    "fuse": 50,
    "rerank": 1_000,
    "gate": 100,
    "generate_first_token": 1_200,
    "generate_complete": 10_000,
    "gate_form": 50,
    "gate_literal": 100,
    "gate_entail": 500,
    "gate_coverage": 50,
    "gate_conflict": 50,
}

DEFAULT_HARD_MS: dict[str, float] = {
    "parse": 120_000,
    "normalize": 30_000,
    "chunk": 120_000,
    "contextualize": 300_000,
    "embed": 300_000,
    "embed_query": 5_000,
    "dense": 2_000,
    "lexical": 2_000,
    "fuse": 1_000,
    "rerank": 30_000,
    "gate": 2_000,
    "generate_first_token": 60_000,
    "generate_complete": 180_000,
    "gate_form": 1_000,
    "gate_literal": 2_000,
    "gate_entail": 10_000,
    "gate_coverage": 1_000,
    "gate_conflict": 1_000,
}

_CONTENT_KEYS = frozenset({"query", "text", "passage", "passages", "answer", "content", "prompt"})


class StageBudgetExceeded(TimeoutError):
    """Hard stage budget blown — callers should enter the §10 degradation path."""

    def __init__(self, stage: str, budget_ms: float, elapsed_ms: float):
        self.stage = stage
        self.budget_ms = budget_ms
        self.elapsed_ms = elapsed_ms
        super().__init__(f"stage {stage} exceeded hard budget {budget_ms:.0f}ms ({elapsed_ms:.0f}ms)")


def configure_logging(fmt: str | None = None) -> None:
    """Set json (default) or console log format. Also honours LOG_FORMAT env."""
    global _log_format
    if fmt is not None:
        _log_format = fmt.strip().lower()
    else:
        _log_format = os.environ.get("LOG_FORMAT", "json").strip().lower()
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger("rag")
    root.setLevel(level)
    for name, value in os.environ.items():
        if name.startswith("LOG_LEVEL_") and name != "LOG_LEVEL":
            mod = name[len("LOG_LEVEL_") :].replace("_", ".").lower()
            if not mod.startswith("rag"):
                mod = "rag." + mod
            logging.getLogger(mod).setLevel(getattr(logging, value.upper(), level))


def content_logging_enabled() -> bool:
    return os.environ.get("LOG_CONTENT", "").strip() in {"1", "true", "yes"}


def set_stage_bus(bus) -> None:
    """Attach a queue-like bus (``put(dict)``) so span transitions can feed SSE."""
    _stage_bus_local.bus = bus


def clear_stage_bus() -> None:
    _stage_bus_local.bus = None


def _publish_stage(record: dict[str, Any]) -> None:
    bus = getattr(_stage_bus_local, "bus", None)
    if bus is None:
        return
    event = record.get("event")
    if event not in {"start", "finish", "error", "stage_slow"}:
        return
    payload = {
        "trace_id": record.get("trace_id"),
        "stage": record.get("stage"),
        "event": event,
        "duration_ms": record.get("duration_ms"),
        "candidates_in": record.get("candidates_in"),
        "candidates_out": record.get("candidates_out"),
        "top_score": record.get("top_score"),
        "model": record.get("model"),
        "cache_hit": record.get("cache_hit"),
        "degraded": record.get("degraded"),
        "skipped": record.get("skipped"),
        "reason": record.get("reason"),
        "error_type": record.get("error_type"),
        "error": record.get("error"),
        "soft_budget_ms": record.get("soft_budget_ms"),
    }
    try:
        bus.put(payload)
    except Exception:
        pass


def current_trace_id() -> str | None:
    return _trace_id.get() or getattr(_trace_local, "trace_id", None)


def set_trace_id(trace_id: str | None):
    _trace_local.trace_id = trace_id
    return _trace_id.set(trace_id)


def reset_trace_id(token) -> None:
    _trace_local.trace_id = None
    _trace_id.reset(token)


@contextmanager
def bind_trace(trace_id: str) -> Iterator[str]:
    token = set_trace_id(trace_id)
    try:
        yield trace_id
    finally:
        reset_trace_id(token)


def uuid7() -> uuid.UUID:
    """RFC 9562 UUIDv7 (time-ordered). Stdlib until 3.13 lacks uuid.uuid7."""
    if hasattr(uuid, "uuid7"):
        return uuid.uuid7()  # type: ignore[attr-defined]
    unix_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    value = (unix_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return uuid.UUID(int=value)


def mint_trace_id(inbound: str | None = None) -> str:
    if inbound and inbound.strip():
        return inbound.strip()[:128]
    return str(uuid7())


def new_request_id() -> str:
    """Backward-compatible short id; prefer mint_trace_id for new code."""
    return secrets.token_hex(6)


def soft_budget_ms(stage: str) -> float:
    env = os.environ.get(f"STAGE_SOFT_MS_{stage}")
    if env:
        return float(env)
    return float(DEFAULT_SOFT_MS.get(stage, 5_000))


def hard_budget_ms(stage: str) -> float:
    env = os.environ.get(f"STAGE_HARD_MS_{stage}")
    if env:
        return float(env)
    return float(DEFAULT_HARD_MS.get(stage, 60_000))


def percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return float(samples[0])
    ordered = sorted(samples)
    rank = max(1, min(len(ordered), int(round(p / 100.0 * len(ordered)))))
    return float(ordered[rank - 1])


def _scrub(fields: dict[str, Any]) -> dict[str, Any]:
    if content_logging_enabled():
        return fields
    return {key: value for key, value in fields.items() if key not in _CONTENT_KEYS}


def _emit(record: dict[str, Any]) -> None:
    if _log_format == "console":
        stage = record.get("stage", "")
        event = record.get("event", "")
        trace = record.get("trace_id") or "-"
        duration = record.get("duration_ms")
        dur = f" {duration:.1f}ms" if isinstance(duration, (int, float)) else ""
        extras = {
            key: value
            for key, value in record.items()
            if key not in {"ts", "trace_id", "stage", "event", "duration_ms"}
        }
        extra = (" " + json.dumps(extras, ensure_ascii=False, default=str)) if extras else ""
        line = f"{record.get('ts', '')} {trace} {stage} {event}{dur}{extra}\n"
        sys.stdout.write(line)
    else:
        sys.stdout.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def log_event(event: str, **fields) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **_scrub(fields),
    }
    trace = current_trace_id()
    if trace and "trace_id" not in payload:
        payload["trace_id"] = trace
    # Keep request_id alias when callers still pass it.
    if "request_id" in fields and "trace_id" not in payload:
        payload["trace_id"] = fields["request_id"]
    _emit(payload)


def log_span(
    stage: str,
    event: str,
    *,
    duration_ms: float | None = None,
    error: BaseException | None = None,
    **payload,
) -> None:
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": current_trace_id(),
        "stage": stage,
        "event": event,
    }
    if duration_ms is not None:
        record["duration_ms"] = round(duration_ms, 3)
    scrubbed = _scrub(payload)
    for key in (
        "candidates_in",
        "candidates_out",
        "top_score",
        "model",
        "cache_hit",
        "degraded",
        "file",
        "file_index",
        "file_total",
        "page",
        "page_total",
        "chunks",
        "layout_escalations",
        "ocr_pages",
        "cache_hit_rate",
        "skipped",
        "reason",
    ):
        if key in scrubbed:
            record[key] = scrubbed[key]
    for key, value in scrubbed.items():
        if key not in record:
            record[key] = value
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
    _emit(record)
    _publish_stage(record)


class StageTimer:
    def __init__(self, clock: Callable[[], float] | None = None):
        self._clock = clock or time.perf_counter
        self._ms: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str, **payload) -> Iterator[dict]:
        # Alias legacy "embed" timer name to the span name embed_query.
        stage = "embed_query" if name == "embed" else name
        started = self._clock()
        with span(stage, **payload) as handle:
            try:
                yield handle
            finally:
                self._ms[name] = (self._clock() - started) * 1000.0

    def as_ms(self) -> dict[str, float]:
        return dict(self._ms)

    def get(self, name: str, default: float = 0.0) -> float:
        return self._ms.get(name, default)


@contextmanager
def span(stage: str, **payload) -> Iterator[dict[str, Any]]:
    """Emit start → finish|error. Soft budget → stage_slow while running."""
    handle: dict[str, Any] = dict(payload)
    soft = soft_budget_ms(stage)
    hard = hard_budget_ms(stage)
    stop = threading.Event()
    started = time.perf_counter()
    trace = current_trace_id()
    bus = getattr(_stage_bus_local, "bus", None)
    public = {key: value for key, value in handle.items() if not str(key).startswith("_")}
    log_span(stage, "start", **public)

    def _watch() -> None:
        # Watcher is another thread — pin request locals explicitly.
        if trace:
            _trace_local.trace_id = trace
        if bus is not None:
            _stage_bus_local.bus = bus
        if stop.wait(max(soft, 0.0) / 1000.0):
            return
        elapsed = (time.perf_counter() - started) * 1000.0
        log_span(stage, "stage_slow", duration_ms=elapsed, soft_budget_ms=soft, **public)
        remaining = max(0.0, hard - soft)
        if remaining <= 0 or stop.wait(remaining / 1000.0):
            return
        handle["_hard_exceeded"] = True
        handle["_hard_elapsed_ms"] = (time.perf_counter() - started) * 1000.0

    threading.Thread(target=_watch, name=f"stage-watch-{stage}", daemon=True).start()
    error: BaseException | None = None
    try:
        yield handle
        if handle.get("_hard_exceeded"):
            raise StageBudgetExceeded(
                stage, hard, float(handle.get("_hard_elapsed_ms") or (time.perf_counter() - started) * 1000.0)
            )
        elapsed_ok = (time.perf_counter() - started) * 1000.0
        if elapsed_ok > hard:
            raise StageBudgetExceeded(stage, hard, elapsed_ok)
    except BaseException as err:
        error = err
        raise
    finally:
        stop.set()
        elapsed = (time.perf_counter() - started) * 1000.0
        merged = {key: value for key, value in handle.items() if not str(key).startswith("_")}
        log_span(
            stage,
            "error" if error is not None else "finish",
            duration_ms=elapsed,
            error=error,
            **merged,
        )
