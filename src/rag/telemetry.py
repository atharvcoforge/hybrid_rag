"""Per-request stage timing and structured logs.

Stages are named for the report columns, not for every internal call.
A missing stage means that arm did not run for this request.
"""

from __future__ import annotations

import json
import secrets
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Iterator


def percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return float(samples[0])
    ordered = sorted(samples)
    # Nearest-rank: p=50 on 1..100 → 50, p=95 → 95, p=99 → 99.
    rank = max(1, min(len(ordered), int(round(p / 100.0 * len(ordered)))))
    return float(ordered[rank - 1])


def new_request_id() -> str:
    return secrets.token_hex(6)


class StageTimer:
    def __init__(self, clock: Callable[[], float] | None = None):
        self._clock = clock or time.perf_counter
        self._ms: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = self._clock()
        try:
            yield
        finally:
            self._ms[name] = (self._clock() - started) * 1000.0

    def as_ms(self) -> dict[str, float]:
        return dict(self._ms)

    def get(self, name: str, default: float = 0.0) -> float:
        return self._ms.get(name, default)


def log_event(event: str, **fields) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()
