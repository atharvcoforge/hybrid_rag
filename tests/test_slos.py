"""Eval SLO gates (§08 latency)."""

from rag.evaluate import Score, check_slos


def test_check_slos_flags_slow_retrieve_p95():
    scores = [
        Score(mode="cascade", kind="all", recall=1.0, mrr=1.0, abstain=0.0, n=10, p95=500.0),
    ]
    failures = check_slos(scores)
    assert any("retrieve p95" in f for f in failures)


def test_check_slos_ignores_a_slow_arm_that_is_not_live():
    scores = [
        Score(mode="bm25", kind="all", recall=1.0, mrr=1.0, abstain=0.0, n=10, p95=20.0),
        Score(mode="rerank", kind="all", recall=1.0, mrr=1.0, abstain=0.0, n=10, p95=900.0),
    ]
    assert check_slos(scores, live_mode="bm25") == []


def test_check_slos_passes_warm_cascade():
    scores = [
        Score(mode="cascade", kind="all", recall=1.0, mrr=1.0, abstain=0.0, n=10, p50=80.0, p95=200.0),
        Score(mode="rrf", kind="all", recall=0.9, mrr=0.8, abstain=0.0, n=10, p95=150.0),
    ]
    assert check_slos(scores) == []
