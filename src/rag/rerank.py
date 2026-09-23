"""Swappable rerankers.

Default is a local cross-encoder. Null keeps RRF order. Jev is a documented
stub that refuses unless both a key and ALLOW_REMOTE_INFERENCE=1 are set —
policy text must not leave the machine by accident.
"""

from __future__ import annotations

import os
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


class JevReranker:
    """Hosted typed-judgment API. Local-by-default policy refuses it."""

    calibrated = True

    def score(self, query: str, passages: list[str]) -> list[float]:
        del query, passages
        if os.environ.get("ALLOW_REMOTE_INFERENCE") != "1":
            raise RuntimeError("JevReranker blocked: remote inference is not allowed")
        if not os.environ.get("JEV_API_KEY"):
            raise RuntimeError("JevReranker needs JEV_API_KEY")
        raise RuntimeError("JevReranker is a stub — wire the client when remote is approved")


# Ablation model ids recorded for eval rows; swapped via config, not imports.
ABLATION_RERANKERS = (
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
    "mixedbread-ai/mxbai-rerank-base-v2",
)
