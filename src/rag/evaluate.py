import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from rag.models import TAU_KEEP, tau_key
from rag.telemetry import percentile

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
    p95: float = 0.0
    p99: float = 0.0
    groundedness: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    answerable_abstain: float | None = None
    unanswerable_abstain: float | None = None


def load_rows(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_split(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_for_split(rows: list[dict], split: dict, which: str) -> list[dict]:
    wanted = set(split[which])
    return [row for row in rows if row["id"] in wanted]


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
    return percentile(samples, 50)


_PREFER = {"rerank": 0, "rrf": 1, "cascade": 2, "dense": 3, "bm25": 4}


def pick_live(lines: list[Score]) -> str:
    overall = [line for line in lines if line.kind == "all"]
    best = max(line.recall for line in overall)
    eligible = [line for line in overall if line.recall + 0.05 >= best]
    fastest = min(line.p50 for line in eligible)
    near = [line for line in eligible if line.p50 <= fastest * 1.25 + 50]
    near.sort(key=lambda line: (-line.mrr, line.p50, _PREFER.get(line.mode, 9)))
    return near[0].mode


def evaluate(index, rows, ask, verify=None) -> tuple[list[Score], float | None]:
    results = {}
    timings: dict[str, list[tuple[str, float]]] = {}
    verified: dict[str, list] = {}

    def clocked(mode, row):
        from rag.embed import reset_caches

        reset_caches()
        started = time.perf_counter()
        result = ask(mode, row)
        elapsed = (time.perf_counter() - started) * 1000
        if result.stages_ms and "total" in result.stages_ms:
            elapsed = result.stages_ms["total"]
        timings.setdefault(mode, []).append((row.get("kind", ""), elapsed))
        if verify is not None:
            verified.setdefault(mode, []).append(verify(row, result))
        return result

    for mode in MODES:
        if mode == "cascade":
            continue
        results[mode] = [clocked(mode, row) for row in rows]
    fitted_by_mode = {}
    for mode in ("dense", "bm25", "rrf", "rerank"):
        rank1 = []
        for row, result in zip(rows, results[mode]):
            if result.hits and row_hit(result.hits[:1], row, k=1):
                rank1.append(result.hits[0].score)
        fitted = fit_tau(rank1)
        if fitted is not None:
            index.set_tau(tau_key(index.model_id, index.model_revision, mode), fitted)
            fitted_by_mode[mode] = fitted
    results["cascade"] = [clocked("cascade", row) for row in rows]
    cascade_rank1 = []
    for row, result in zip(rows, results["cascade"]):
        if result.hits and row_hit(result.hits[:1], row, k=1):
            cascade_rank1.append(result.hits[0].score)
    cascade_fitted = fit_tau(cascade_rank1)
    if cascade_fitted is not None:
        index.set_tau(tau_key(index.model_id, index.model_revision, "cascade"), cascade_fitted)
        fitted_by_mode["cascade"] = cascade_fitted
    fitted = fitted_by_mode.get("rerank")
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
            answerable = [(row, result) for row, result in chosen if row.get("must_contain")]
            unanswerable = [
                (row, result) for row, result in chosen if row.get("kind") == "unanswerable"
            ]
            gate_scores = None
            if verify is not None and kind == "all":
                gate_scores = verified.get(mode) or []
            lines.append(
                Score(
                    mode=mode,
                    kind=kind,
                    recall=(
                        sum(row_hit(result.hits, row) for row, result in graded) / len(graded)
                        if graded
                        else 0.0
                    ),
                    mrr=(
                        sum(reciprocal(result.hits, row) for row, result in graded) / len(graded)
                        if graded
                        else 0.0
                    ),
                    abstain=sum(result.reason == "no_confident_hit" for _row, result in chosen) / n,
                    n=n,
                    p50=percentile(taken, 50),
                    p95=percentile(taken, 95),
                    p99=percentile(taken, 99),
                    groundedness=_mean_key(gate_scores, "groundedness") if gate_scores is not None else None,
                    citation_precision=_mean_key(gate_scores, "citation_precision") if gate_scores is not None else None,
                    citation_recall=_mean_key(gate_scores, "citation_recall") if gate_scores is not None else None,
                    answerable_abstain=(
                        sum(result.reason == "no_confident_hit" for _row, result in answerable) / len(answerable)
                        if kind == "all" and answerable
                        else None
                    ),
                    unanswerable_abstain=(
                        sum(result.reason == "no_confident_hit" for _row, result in unanswerable) / len(unanswerable)
                        if kind == "all" and unanswerable
                        else None
                    ),
                )
            )
    return lines, fitted


def _mean_key(rows: list | None, key: str) -> float | None:
    if not rows:
        return None
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def format_scores(lines: list[Score], fitted) -> str:
    rendered = []
    for line in lines:
        base = (
            f"{line.mode:10} {line.kind:12} recall@5={line.recall:.2f} mrr={line.mrr:.2f} "
            f"abstain={line.abstain:.2f} p50={line.p50:.0f}ms p95={line.p95:.0f}ms p99={line.p99:.0f}ms n={line.n}"
        )
        if line.kind == "all" and line.answerable_abstain is not None:
            base += (
                f" ans_abs={line.answerable_abstain:.2f} unans_abs={line.unanswerable_abstain:.2f}"
            )
        if line.groundedness is not None:
            base += (
                f" ground={line.groundedness:.2f} cite_p={line.citation_precision:.2f} "
                f"cite_r={line.citation_recall:.2f}"
            )
        rendered.append(base)
    rendered.append("tau unset" if fitted is None else f"tau={fitted:.4f}")
    rendered.append(f"live={pick_live(lines)}")
    return "\n".join(rendered)


# Warm retrieve budget from §08. Eval asserts these; CI fails a regression.
RETRIEVE_P95_MS = 400.0
TTFT_P95_MS = 1200.0


def check_slos(lines: list[Score], *, ttft_p95: float | None = None) -> list[str]:
    """Return human-readable SLO failures. Empty list means green."""
    failures = []
    for line in lines:
        if line.mode in ("cascade", "rrf", "rerank") and line.kind == "all" and line.p95 > RETRIEVE_P95_MS:
            failures.append(
                f"retrieve p95 {line.p95:.0f}ms exceeds {RETRIEVE_P95_MS:.0f}ms ({line.mode})"
            )
    if ttft_p95 is not None and ttft_p95 > TTFT_P95_MS:
        failures.append(f"ttft p95 {ttft_p95:.0f}ms exceeds {TTFT_P95_MS:.0f}ms")
    return failures
