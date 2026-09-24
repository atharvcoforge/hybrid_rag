"""Swappable rerankers.

Default is a local cross-encoder. Null keeps the incoming order when the
cross-encoder fails to load.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    calibrated: bool

    def score(self, query: str, passages: list[str]) -> list[float]:
        ...


class NullReranker:
    """Fallback when the cross-encoder fails to load or times out."""

    calibrated = False

    def score(self, query: str, passages: list[str]) -> list[float]:
        del query
        n = len(passages)
        return [float(n - i) for i in range(n)]


class CrossEncoderReranker:
    calibrated = False

    def __init__(self, model: Any = None, *, max_length: int = 256) -> None:
        self._model = model
        self.max_length = max_length

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        from rag.embed import load_reranker

        self._model = load_reranker()
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        model = self._load()
        pairs = [(query, text[: self.max_length * 4]) for text in passages]
        scores = model.predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]


