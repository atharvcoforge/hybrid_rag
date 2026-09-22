"""Reranker protocol: swappable scorers (§09)."""

import os

import pytest

from rag.rerank import CrossEncoderReranker, JevReranker, NullReranker


def test_null_reranker_preserves_input_order_with_descending_scores():
    reranker = NullReranker()
    scores = reranker.score("q", ["a", "b", "c"])
    assert scores == [3.0, 2.0, 1.0]
    assert reranker.calibrated is False


def test_cross_encoder_reranker_delegates_to_predict():
    class FakeModel:
        def predict(self, pairs, show_progress_bar=False):
            del show_progress_bar
            return [0.1 * i for i, _ in enumerate(pairs)]

    reranker = CrossEncoderReranker(model=FakeModel())
    assert reranker.score("q", ["x", "y"]) == [0.0, 0.1]
    assert reranker.calibrated is False


def test_jev_reranker_is_blocked_without_explicit_remote_allowance(monkeypatch):
    monkeypatch.delenv("ALLOW_REMOTE_INFERENCE", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    reranker = JevReranker()
    with pytest.raises(RuntimeError, match="remote"):
        reranker.score("q", ["a"])


def test_jev_reranker_still_raises_without_a_key_even_when_allowed(monkeypatch):
    monkeypatch.setenv("ALLOW_REMOTE_INFERENCE", "1")
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    reranker = JevReranker()
    with pytest.raises(RuntimeError, match="JEV_API_KEY"):
        reranker.score("q", ["a"])
