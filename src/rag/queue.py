"""Bounded inference admission (F-23)."""

from __future__ import annotations

import threading
from typing import Literal, Self


class BusyError(Exception):
    """Queue is full — caller should return 429."""

    def __init__(self, retry_after: int = 2) -> None:
        self.retry_after = retry_after
        super().__init__("inference queue full")


class InferenceQueue:
    def __init__(self, maxsize: int = 8) -> None:
        self.maxsize = maxsize
        self._sem = threading.BoundedSemaphore(maxsize)
        self.in_flight = 0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if not self._sem.acquire(blocking=False):
            raise BusyError()
        with self._lock:
            self.in_flight += 1

    def release(self) -> None:
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
        self._sem.release()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> Literal[False]:
        self.release()
        return False
