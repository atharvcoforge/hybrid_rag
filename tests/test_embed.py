from rag.embed import embed_texts
from tests.fakes import fake_encode


def test_unchanged_text_is_not_encoded_twice():
    cache = {}

    def cache_get(model_id, revision, digest):
        return cache.get((model_id, revision, digest))

    def cache_put(model_id, revision, digest, vector):
        cache[(model_id, revision, digest)] = vector

    calls = []

    def encode(texts, *, query=False):
        calls.append(list(texts))
        return fake_encode(texts, query=query)

    first = embed_texts(["same passage"], encode, cache_get, cache_put, "m", "r")
    second = embed_texts(["same passage"], encode, cache_get, cache_put, "m", "r")
    assert first == second
    assert calls == [["same passage"]]


def test_query_flag_is_forwarded():
    seen = {}

    def encode(texts, *, query=False):
        seen["query"] = query
        return fake_encode(texts, query=query)

    embed_texts(["passage"], encode)
    assert seen["query"] is False
