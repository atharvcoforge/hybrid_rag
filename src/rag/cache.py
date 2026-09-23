"""Bounded, generation-keyed answer cache (F-22)."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

CacheKey = tuple[str, int, str, str]


class AnswerCache:
    def __init__(self, maxsize: int = 256, ttl_s: float = 3600.0) -> None:
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._data: OrderedDict[CacheKey, tuple[object, float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(query: str, generation: int, mode: str, doc_id: str | None = None) -> CacheKey:
        return (query.casefold(), int(generation), mode, (doc_id or "").casefold())

    def get(self, key: CacheKey) -> object | None:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            value, stamped = item
            if now - stamped > self.ttl_s:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def __setitem__(self, key: CacheKey, value: object) -> None:
        with self._lock:
            self._data[key] = (value, time.monotonic())
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
