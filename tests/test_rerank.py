"""Reranker protocol: swappable scorers."""

from rag.rerank import CrossEncoderReranker, NullReranker


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

