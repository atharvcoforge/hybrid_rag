import json
import math
from dataclasses import dataclass
from pathlib import Path

from rag.models import TAU_KEEP, tau_key

MODES = ("dense", "bm25", "rrf", "rerank", "cascade")


@dataclass
class Score:
    mode: str
    kind: str
    recall: float
    mrr: float
    abstain: float
    n: int


def load_rows(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def fit_tau(scores: list[float], keep: float = TAU_KEEP) -> float | None:
    if not scores:
        return None
    ordered = sorted(scores)
    keep_n = math.ceil(keep * len(ordered))
    return ordered[len(ordered) - keep_n]


def row_hit(hits, row, k: int = 5) -> bool:
    needle = row["must_contain"]
    doc_id = row["doc_id"]
    for hit in hits[:k]:
        if needle in hit.parent_text and hit.source_path == doc_id:
            return True
    return False


def reciprocal(hits, row) -> float:
    needle = row["must_contain"]
    doc_id = row["doc_id"]
    for index, hit in enumerate(hits, start=1):
        if needle in hit.parent_text and hit.source_path == doc_id:
            return 1.0 / index
    return 0.0


def evaluate(index, rows, ask) -> tuple[list[Score], float | None]:
    results = {}
    for mode in MODES:
        if mode == "cascade":
            continue
        results[mode] = [ask(mode, row) for row in rows]
    rank1 = []
    for row, result in zip(rows, results["rerank"]):
        if result.hits and row_hit(result.hits[:1], row, k=1):
            rank1.append(result.hits[0].score)
    fitted = fit_tau(rank1)
    if fitted is not None:
        index.set_tau(tau_key(index.model_id, index.model_revision), fitted)
    results["cascade"] = [ask("cascade", row) for row in rows]
    lines = []
    kinds = ["all", *sorted({row["kind"] for row in rows})]
    for mode in MODES:
        for kind in kinds:
            chosen = [
                (row, result)
                for row, result in zip(rows, results[mode])
                if kind == "all" or row["kind"] == kind
            ]
            if not chosen:
                continue
            n = len(chosen)
            lines.append(
                Score(
                    mode=mode,
                    kind=kind,
                    recall=sum(row_hit(result.hits, row) for row, result in chosen) / n,
                    mrr=sum(reciprocal(result.hits, row) for row, result in chosen) / n,
                    abstain=sum(result.reason == "no_confident_hit" for _row, result in chosen) / n,
                    n=n,
                )
            )
    return lines, fitted


def format_scores(lines: list[Score], fitted) -> str:
    rendered = [
        f"{line.mode:10} {line.kind:10} recall@5={line.recall:.2f} mrr={line.mrr:.2f} abstain={line.abstain:.2f} n={line.n}"
        for line in lines
    ]
    rendered.append("tau unset" if fitted is None else f"tau={fitted:.4f}")
    return "\n".join(rendered)
