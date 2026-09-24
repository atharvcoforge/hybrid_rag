import hashlib
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from rag.models import (
    EMBED_BATCH,
    EMBED_MODEL,
    EMBED_REVISION,
    RERANK_MODEL,
    RERANK_REVISION,
    IngestError,
)

_embedder: Any = None
_reranker: Any = None


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embed_texts(
    texts: Sequence[str],
    encode: Callable[..., Sequence[Sequence[float]]],
    cache_get: Callable[[str, str, str], list[float] | None] | None = None,
    cache_put: Callable[[str, str, str, list[float]], None] | None = None,
    model_id: str = "",
    revision: str = "",
) -> list[list[float]]:
    vectors: list[list[float] | None] = [None] * len(texts)
    missing: list[int] = []
    for index, text in enumerate(texts):
        cached = None
        if cache_get is not None:
            cached = cache_get(model_id, revision, text_hash(text))
        if cached is None:
            missing.append(index)
        else:
            vectors[index] = cached
    if missing:
        fresh = encode([texts[index] for index in missing], query=False)
        if len(fresh) != len(missing):
            raise IngestError("", "encoder returned the wrong number of vectors")
        for index, row in zip(missing, fresh):
            stored = [float(value) for value in row]
            vectors[index] = stored
            if cache_put is not None:
                cache_put(model_id, revision, text_hash(texts[index]), stored)
    done: list[list[float]] = []
    for vector in vectors:
        if vector is None:
            raise IngestError("", "encoder left a hole")
        done.append(vector)
    return done


def encode_documents(texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
    if query:
        return [encode_query(text) for text in texts]
    model = load_embedder()
    encoded = model.encode(
        list(texts),
        normalize_embeddings=True,
        batch_size=EMBED_BATCH,
        show_progress_bar=False,
    )
    return [[float(value) for value in row] for row in encoded]


@lru_cache(maxsize=128)
def _cached_query(model_id: str, revision: str, text: str) -> tuple[float, ...]:
    model = load_embedder()
    encoded = model.encode(
        [text],
        prompt_name="query",
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return tuple(float(value) for value in encoded[0])


def encode_query(text: str) -> list[float]:
    return list(_cached_query(EMBED_MODEL, EMBED_REVISION, text))


def reset_caches() -> None:
    _cached_query.cache_clear()


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(load_embedder().tokenizer.encode(text, add_special_tokens=False))


def load_embedder() -> Any:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(
            EMBED_MODEL,
            revision=EMBED_REVISION,
            device=_device(),
        )
    return _embedder


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    from rag.rerank import CrossEncoderReranker

    return CrossEncoderReranker().score(query, texts)


def load_reranker() -> Any:
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(
            RERANK_MODEL, revision=RERANK_REVISION, max_length=512, device=_device()
        )
    return _reranker


def _device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 — torch import/backend failures are an open set
        return "cpu"
    return "cpu"
