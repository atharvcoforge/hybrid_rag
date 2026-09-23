"""Branch coverage for rag.embed. SentenceTransformer is stubbed."""

import ctypes
import inspect
import sys
import types

import pytest

from rag.embed import (
    _device,
    count_tokens,
    embed_texts,
    encode_documents,
    encode_query,
    load_embedder,
    load_reranker,
    rerank_scores,
    reset_caches,
)
from rag.models import IngestError


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class _Model:
    def __init__(self, *args, **kwargs):
        self.tokenizer = _Tok()

    def encode(self, texts, **kwargs):
        return [[0.6, 0.8] for _ in list(texts)]


class _Cross:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs, show_progress_bar=False):
        return [0.5 for _ in pairs]


def _install_stub(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _Model
    module.CrossEncoder = _Cross
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_stubbed_embedder_query_path_and_reranker(monkeypatch):
    from rag import embed

    embed._embedder = None
    embed._reranker = None
    reset_caches()
    _install_stub(monkeypatch)
    try:
        first = load_embedder()
        assert load_embedder() is first
        assert count_tokens("") == 0
        assert count_tokens("one two") == 2
        vectors = encode_documents(["policy"], query=False)
        assert vectors == [[0.6, 0.8]]
        queried = encode_documents(["policy"], query=True)
        assert queried == [encode_query("policy")]
        reranker = load_reranker()
        assert load_reranker() is reranker
        assert rerank_scores("q", []) == []
    finally:
        embed._embedder = None
        embed._reranker = None
        reset_caches()


def test_device_cpu_when_mps_missing_or_torch_fails(monkeypatch):
    torch = types.ModuleType("torch")
    backends = types.ModuleType("torch.backends")
    mps = types.ModuleType("torch.backends.mps")
    mps.is_available = lambda: False
    backends.mps = mps
    torch.backends = backends
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert _device() == "cpu"
    mps.is_available = lambda: True
    assert _device() == "mps"

    monkeypatch.setitem(sys.modules, "torch", None)
    assert _device() == "cpu"


def test_embed_texts_cache_mismatch_and_hole():
    cache = {}

    def cache_get(_model, _rev, digest):
        return cache.get(digest)

    def cache_put(_model, _rev, digest, vector):
        cache[digest] = vector

    def encode(texts, query=False):
        return [[1.0, 0.0] for _ in texts]

    first = embed_texts(["a", "b"], encode, cache_get, cache_put, "m", "r")
    second = embed_texts(["a", "b"], encode, cache_get, cache_put, "m", "r")
    assert first == second

    def wrong(texts, query=False):
        return []

    with pytest.raises(IngestError, match="wrong number"):
        embed_texts(["a"], wrong)

    def hole(texts, query=False):
        frame = inspect.currentframe().f_back

        class Hole(list):
            def __setitem__(self, index, value):
                list.__setitem__(self, index, None)

        frame.f_locals["vectors"] = Hole([None] * len(texts))
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))
        return [[1.0] for _ in texts]

    with pytest.raises(IngestError, match="hole"):
        embed_texts(["a"], hole)
