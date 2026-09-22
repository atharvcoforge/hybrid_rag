"""Fit confidence thresholds without scoring side effects.

`rag calibrate` writes. `rag eval` only reads. Fitting includes negatives and
uses a held-out split when one is provided.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.evaluate import fit_tau, load_rows, load_split, row_hit, rows_for_split
from rag.models import tau_key


def fit_mode_threshold(rows, results, keep: float = 0.95) -> float | None:
    """Fit on rank-1 scores of answerable hits and unanswerable tops.

    Positives: correct rank-1 scores. Negatives: top score on unanswerable
    rows (and wrong answerable hits). Threshold maximises coverage at the
    keep-precision operating point on the combined list — for now we reuse
    fit_tau on positives only when negatives are empty, otherwise take the
    midpoint between the lowest kept positive and highest negative.
    """
    positives = []
    negatives = []
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
    # Precision-oriented: threshold just above the highest negative that still
    # keeps `keep` of the positives.
    ordered = sorted(positives)
    keep_n = max(1, int(round(keep * len(ordered))))
    candidate = ordered[len(ordered) - keep_n]
    floor = max(negatives)
    return max(candidate, floor)


def calibrate(index, rows, ask, split=None, modes=("rrf", "rerank", "cascade")) -> dict:
    if split is not None:
        train_rows = rows_for_split(rows, split, "train")
        held_rows = rows_for_split(rows, split, "test")
    else:
        train_rows = rows
        held_rows = []
    report = {"modes": {}, "held_out": {}}
    for mode in modes:
        train_results = [ask(mode, row) for row in train_rows]
        fitted = fit_mode_threshold(train_rows, train_results)
        if fitted is None:
            report["modes"][mode] = {"tau": None}
            continue
        key = tau_key(index.model_id, index.model_revision, mode)
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


def write_report(path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
