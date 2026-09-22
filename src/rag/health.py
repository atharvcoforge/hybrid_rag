"""Process health and generator circuit breaker (§10)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class HealthState:
    dense_ok: bool = True
    fts_ok: bool = True
    rerank_ok: bool = True
    nli_ok: bool = True
    writer_ok: bool = True
    warmed: bool = False
    index_generation: int = 0
    messages: list[str] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "dense_ok": self.dense_ok,
            "fts_ok": self.fts_ok,
            "rerank_ok": self.rerank_ok,
            "nli_ok": self.nli_ok,
            "writer_ok": self.writer_ok,
            "warmed": self.warmed,
            "index_generation": self.index_generation,
            "degraded": self.degraded(),
            "messages": list(self.messages),
        }

    def degraded(self) -> bool:
        return not all(
            (self.dense_ok, self.fts_ok, self.rerank_ok, self.nli_ok, self.writer_ok)
        )

    def note(self, message: str) -> None:
        if message and message not in self.messages:
            self.messages.append(message)


class CircuitBreaker:
    """Opens after consecutive generator failures; half-opens after reset_s."""

    def __init__(self, fail_threshold: int = 3, reset_s: float = 60.0):
        self.fail_threshold = fail_threshold
        self.reset_s = reset_s
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            if time.monotonic() - self.opened_at >= self.reset_s:
                return True  # half-open probe
            return False

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.fail_threshold:
                self.opened_at = time.monotonic()

    @property
    def state(self) -> str:
        with self._lock:
            if self.opened_at is None:
                return "closed"
            if time.monotonic() - self.opened_at >= self.reset_s:
                return "half_open"
            return "open"
