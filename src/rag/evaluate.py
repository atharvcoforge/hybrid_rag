import json
import math
import time
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
    p50: float = 0.0


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
    needle = row.get("must_contain") or ""
    doc_id = row.get("doc_id")
    if not needle or not doc_id:
        return False
    for hit in hits[:k]:
        if needle in hit.parent_text and hit.source_path == doc_id:
            return True
    return False


def reciprocal(hits, row) -> float:
    needle = row.get("must_contain") or ""
    doc_id = row.get("doc_id")
    if not needle or not doc_id:
        return 0.0
    for index, hit in enumerate(hits, start=1):
        if needle in hit.parent_text and hit.source_path == doc_id:
            return 1.0 / index
    return 0.0


def percentile_50(samples: list[float]) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    return ordered[(len(ordered) - 1) // 2]


_PREFER = {"rerank": 0, "rrf": 1, "cascade": 2, "dense": 3, "bm25": 4}


def pick_live(lines: list[Score]) -> str:
    overall = [line for line in lines if line.kind == "all"]
    best = max(line.recall for line in overall)
    eligible = [line for line in overall if line.recall + 0.05 >= best]
    fastest = min(line.p50 for line in eligible)
    near = [line for line in eligible if line.p50 <= fastest * 1.25 + 50]
    near.sort(key=lambda line: (-line.mrr, line.p50, _PREFER.get(line.mode, 9)))
    return near[0].mode


def evaluate(index, rows, ask) -> tuple[list[Score], float | None]:
    results = {}
    timings: dict[str, list[tuple[str, float]]] = {}

    def clocked(mode, row):
        from rag.embed import _cached_query

        _cached_query.cache_clear()
        started = time.perf_counter()
        result = ask(mode, row)
        timings.setdefault(mode, []).append((row.get("kind", ""), (time.perf_counter() - started) * 1000))
        return result

    for mode in MODES:
        if mode == "cascade":
            continue
        results[mode] = [clocked(mode, row) for row in rows]
    rank1 = []
    for row, result in zip(rows, results["rerank"]):
        if result.hits and row_hit(result.hits[:1], row, k=1):
            rank1.append(result.hits[0].score)
    fitted = fit_tau(rank1)
    if fitted is not None:
        index.set_tau(tau_key(index.model_id, index.model_revision), fitted)
    results["cascade"] = [clocked("cascade", row) for row in rows]
    lines = []
    kinds = ["all", *sorted({row["kind"] for row in rows})]
    for mode in MODES:
        samples = timings.get(mode, [])
        for kind in kinds:
            chosen = [
                (row, result)
                for row, result in zip(rows, results[mode])
                if kind == "all" or row["kind"] == kind
            ]
            if not chosen:
                continue
            graded = [(row, result) for row, result in chosen if row.get("must_contain")]
            taken = [ms for sample_kind, ms in samples if kind == "all" or sample_kind == kind]
            n = len(chosen)
            lines.append(
                Score(
                    mode=mode,
                    kind=kind,
                    recall=sum(row_hit(result.hits, row) for row, result in graded) / len(graded) if graded else 0.0,
                    mrr=sum(reciprocal(result.hits, row) for row, result in graded) / len(graded) if graded else 0.0,
                    abstain=sum(result.reason == "no_confident_hit" for _row, result in chosen) / n,
                    n=n,
                    p50=percentile_50(taken),
                )
            )
    return lines, fitted


def format_scores(lines: list[Score], fitted) -> str:
    rendered = [
        f"{line.mode:10} {line.kind:10} recall@5={line.recall:.2f} mrr={line.mrr:.2f} abstain={line.abstain:.2f} p50={line.p50:.0f}ms n={line.n}"
        for line in lines
    ]
    rendered.append("tau unset" if fitted is None else f"tau={fitted:.4f}")
    rendered.append(f"live={pick_live(lines)}")
    return "\n".join(rendered)
