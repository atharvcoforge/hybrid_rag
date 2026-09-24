"""Fit confidence thresholds without scoring side effects.

`rag calibrate` writes. `rag eval` only reads. Fitting includes negatives and
uses a held-out split when one is provided.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag.evaluate import fit_tau, row_hit, rows_for_split
from rag.models import Retrieval, tau_key


def fit_mode_threshold(
    rows: list[dict[str, Any]],
    results: list[Retrieval],
    keep: float = 0.95,
) -> float | None:
    """Fit on rank-1 scores of answerable hits and unanswerable tops.

    Positives: correct rank-1 scores. Negatives: top score on unanswerable
    rows (and wrong answerable hits). Threshold maximises coverage at the
    keep-precision operating point on the combined list — for now we reuse
    fit_tau on positives only when negatives are empty, otherwise take the
    midpoint between the lowest kept positive and highest negative.
    """
    positives: list[float] = []
    negatives: list[float] = []
    for row, result in zip(rows, results):
        if not result.hits:
            continue
        top = result.hits[0].score
        if row.get("kind") == "unanswerable":
            negatives.append(top)
        elif row.get("must_contain"):
            if row_hit(result.hits[:1], row, k=1):
                positives.append(top)
            else:
                negatives.append(top)
    if not positives:
        return None
    if not negatives:
        return fit_tau(positives, keep=keep)
    # Lowest threshold whose answered set still has precision >= keep.
    # Sitting above every negative (the old max(candidate, floor)) abstains
    # the positives too when one unanswerable query outscores them.
    labeled = [(score, True) for score in positives] + [(score, False) for score in negatives]
    best_tau: float | None = None
    best_coverage = -1.0
    for tau in sorted({score for score, _label in labeled}):
        answered = [label for score, label in labeled if score >= tau]
        if not answered:
            continue
        precision = sum(answered) / len(answered)
        coverage = sum(score >= tau for score in positives) / len(positives)
        if precision + 1e-12 >= keep and coverage > best_coverage:
            best_coverage = coverage
            best_tau = tau
    if best_tau is not None:
        return best_tau
    return fit_tau(positives, keep=keep)


def calibrate(
    index: Any,
    rows: list[dict[str, Any]],
    ask: Callable[[str, dict[str, Any]], Retrieval],
    split: dict[str, Any] | None = None,
    modes: tuple[str, ...] = ("rrf", "rerank", "cascade"),
) -> dict[str, Any]:
    if split is not None:
        train_rows = rows_for_split(rows, split, "train")
        held_rows = rows_for_split(rows, split, "test")
    else:
        train_rows = rows
        held_rows = []
    report: dict[str, Any] = {"modes": {}, "held_out": {}}
    for mode in modes:
        key = tau_key(index.model_id, index.model_revision, mode)
        # A stored cutoff from a previous score scale would drop every hit
        # before the fit saw it, and the fit could never recover.
        index.set_tau(key, 0.0)
        train_results = [ask(mode, row) for row in train_rows]
        fitted = fit_mode_threshold(train_rows, train_results)
        if fitted is None:
            report["modes"][mode] = {"tau": None}
            continue
        index.set_tau(key, fitted)
        entry = {"tau": fitted, "n": len(train_rows), "key": key}
        if held_rows:
            held_results = [ask(mode, row) for row in held_rows]
            abstain = sum(r.reason == "no_confident_hit" for r in held_results) / len(held_rows)
            unans = [
                (row, result)
                for row, result in zip(held_rows, held_results)
                if row.get("kind") == "unanswerable"
            ]
            unans_abs = (
                sum(result.reason == "no_confident_hit" for _row, result in unans) / len(unans)
                if unans
                else None
            )
            entry["held_out_abstain"] = abstain
            entry["held_out_unanswerable_abstain"] = unans_abs
        report["modes"][mode] = entry
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
