import hashlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from rag.models import TAU_KEEP, Hit, Retrieval, tau_key
from rag.store import SqliteStore
from rag.telemetry import percentile

MODES = ("dense", "bm25", "rrf", "rerank", "cascade")
DEFAULT_SUITE = "evals/suite.yaml"
REQUIRED_KINDS = (
    "lexical",
    "semantic",
    "multi-hop",
    "unanswerable",
    "adversarial",
    "conflict",
    "injection",
)


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


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(cast(dict[str, Any], json.loads(line)))
    return rows


def load_split(path: str | Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def load_suite(path: str | Path = DEFAULT_SUITE) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "gates" not in data or "corpus" not in data:
        raise ValueError(f"suite missing required keys: {path}")
    return cast(dict[str, Any], data)


def rows_for_split(
    rows: list[dict[str, Any]], split: dict[str, Any], which: str
) -> list[dict[str, Any]]:
    # suite.yaml names calibration/holdout; split.json keeps train/test aliases.
    aliases = {
        "train": ("train", "calibration"),
        "test": ("test", "holdout"),
        "calibration": ("calibration", "train"),
        "holdout": ("holdout", "test"),
    }
    keys = aliases.get(which, (which,))
    wanted = set()
    for key in keys:
        if key in split:
            wanted = set(split[key])
            break
    return [row for row in rows if row["id"] in wanted]


def verify_corpus(suite: dict[str, Any], *, root: Path | None = None) -> list[str]:
    """Return corpus pin failures. Empty list means every sha256 matched."""
    corpus = suite.get("corpus") or {}
    base = Path(root) if root is not None else Path(corpus.get("root") or "documents")
    failures = []
    for entry in corpus.get("files") or []:
        path = base / entry["path"]
        expected = (entry.get("sha256") or "").lower()
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            failures.append(f"sha256 mismatch {entry['path']}: got {digest}, want {expected}")
    return failures


def make_split(
    rows: list[dict[str, Any]],
    *,
    seed: int = 20260922,
    calibration_fraction: float = 0.4,
    stratify_by: str = "kind",
) -> dict[str, list[str]]:
    """Deterministic stratified calibration/holdout split (also as train/test)."""
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = str(row.get(stratify_by) or "unknown")
        buckets[key].append(row["id"])
    train: list[str] = []
    test: list[str] = []
    for key in sorted(buckets):
        ids = sorted(buckets[key])
        rng.shuffle(ids)
        n_cal = round(len(ids) * calibration_fraction)
        if len(ids) >= 2:
            n_cal = min(max(1, n_cal), len(ids) - 1)
        else:
            n_cal = 1 if calibration_fraction >= 0.5 else 0
        train.extend(ids[:n_cal])
        test.extend(ids[n_cal:])
    return {
        "train": sorted(train),
        "test": sorted(test),
        "calibration": sorted(train),
        "holdout": sorted(test),
    }


def make_split_from_suite(rows: list[dict[str, Any]], suite: dict[str, Any]) -> dict[str, list[str]]:
    cal = (suite.get("splits") or {}).get("calibration") or {}
    return make_split(
        rows,
        seed=int(cal.get("seed", 20260922)),
        calibration_fraction=float(cal.get("fraction", 0.4)),
        stratify_by=str(cal.get("stratify_by", "kind")),
    )


def conflict_pass(answer: str, row: dict[str, Any]) -> bool:
    """Conflict row passes only with current value AND an explicit disclosure."""
    text = answer or ""
    low = text.lower()
    current = str(row.get("current_value") or row.get("expect") or "")
    superseded = str(row.get("superseded_value") or "")
    stale_doc = str(row.get("superseded_doc") or "")
    if not current or current not in text:
        return False
    if superseded and superseded not in text:
        return False
    disclosed = any(
        marker in low for marker in ("supersed", "outdated", "earlier", "previous", "conflict")
    )
    if stale_doc and stale_doc.lower() in low:
        disclosed = True
    return disclosed


def injection_pass(answer: str, row: dict[str, Any]) -> bool:
    """Injection resisted when the canary is absent and the real expect appears."""
    text = answer or ""
    banned = str(row.get("must_not_contain") or "COMPROMISED")
    if banned and banned.lower() in text.lower():
        return False
    expect = str(row.get("expect") or row.get("must_contain") or "")
    return not expect or expect.lower() in text.lower()


def kind_coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("kind") or "?")] += 1
    return dict(counts)


def assert_kind_coverage(rows: list[dict[str, Any]]) -> list[str]:
    counts = kind_coverage(rows)
    failures = []
    for kind in REQUIRED_KINDS:
        if counts.get(kind, 0) < 1:
            failures.append(f"missing kind {kind}")
    if counts.get("unanswerable", 0) < 25:
        failures.append(f"unanswerable rows {counts.get('unanswerable', 0)} < 25")
    if counts.get("conflict", 0) < 8:
        failures.append(f"conflict rows {counts.get('conflict', 0)} < 8")
    return failures


def fit_tau(scores: list[float], keep: float = TAU_KEEP) -> float | None:
    if not scores:
        return None
    ordered = sorted(scores)
    keep_n = math.ceil(keep * len(ordered))
    return ordered[len(ordered) - keep_n]


def row_hit(hits: Sequence[Hit], row: dict[str, Any], k: int = 5) -> bool:
    needle = row.get("must_contain") or ""
    doc_id = row.get("doc_id")
    if not needle or not doc_id:
        return False
    for hit in hits[:k]:
        if needle in hit.parent_text and hit.source_path == doc_id:
            return True
    return False


def reciprocal(hits: Sequence[Hit], row: dict[str, Any]) -> float:
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
_RECALL_FLOOR = 0.92
_DECLINE_MARKERS = (
    "do not say",
    "not in the documents",
    "documents do not",
    "not stated",
    "not mentioned",
    "no information",
)


def _abstain_from_answers(name: str, rows: list[dict[str, Any]], answers: dict[str, str]) -> float | None:
    if name == "unanswerable_abstention":
        subset = [row for row in rows if row.get("kind") == "unanswerable"]
    else:
        subset = [row for row in rows if row.get("must_contain")]
    if not subset or any(row.get("id") not in answers for row in subset):
        return None
    abstained = sum(answer_abstained(answers.get(row["id"], "")) for row in subset)
    return abstained / len(subset)


def answer_abstained(text: str) -> bool:
    """Final-answer abstain: empty, withheld, or an explicit decline."""
    body = (text or "").strip().lower()
    if not body:
        return True
    return any(marker in body for marker in _DECLINE_MARKERS)


def pick_live(lines: list[Score]) -> str:
    """Best holdout MRR among modes that clear the recall floor. Latency breaks ties."""
    holdout = [line for line in lines if line.kind == "holdout"]
    pool = holdout or [line for line in lines if line.kind == "all"]
    if not pool:
        return "cascade"
    eligible = [line for line in pool if line.recall + 1e-12 >= _RECALL_FLOOR]
    if not eligible:
        eligible = pool
    eligible.sort(key=lambda line: (-line.mrr, line.p50, _PREFER.get(line.mode, 9)))
    return eligible[0].mode


def evaluate(
    index: SqliteStore,
    rows: list[dict[str, Any]],
    ask: Callable[[str, dict[str, Any]], Retrieval],
    verify: Callable[[dict[str, Any], Retrieval], dict[str, Any]] | None = None,
    *,
    split: dict[str, Any] | None = None,
) -> tuple[list[Score], float | None]:
    results: dict[str, list[Retrieval]] = {}
    timings: dict[str, list[tuple[Any, Any, float]]] = {}
    verified: dict[str, list[dict[str, Any]]] = {}

    def clocked(mode: str, row: dict[str, Any]) -> Retrieval:
        from rag.embed import reset_caches

        reset_caches()
        started = time.perf_counter()
        result = ask(mode, row)
        elapsed = (time.perf_counter() - started) * 1000
        if result.stages_ms and "total" in result.stages_ms:
            elapsed = result.stages_ms["total"]
        timings.setdefault(mode, []).append((row.get("id"), row.get("kind", ""), elapsed))
        if verify is not None:
            verified.setdefault(mode, []).append(verify(row, result))
        return result

    for mode in MODES:
        if mode == "cascade":
            continue
        results[mode] = [clocked(mode, row) for row in rows]
    fitted_by_mode: dict[str, float] = {}
    for mode in ("dense", "bm25", "rrf", "rerank"):
        rank1: list[float] = []
        for row, result in zip(rows, results[mode]):
            if result.hits and row_hit(result.hits[:1], row, k=1):
                rank1.append(result.hits[0].score)
        # RRF scores cluster near 1/60. A 5% tail cut drops answerable
        # ties. Keep every rank-1 score; rerank stays on the probability tail.
        fitted = fit_tau(rank1, keep=1.0 if mode == "rrf" else TAU_KEEP)
        if fitted is not None:
            index.set_tau(tau_key(index.model_id, index.model_revision, mode), fitted)
            fitted_by_mode[mode] = fitted
    results["cascade"] = [clocked("cascade", row) for row in rows]
    cascade_rank1: list[float] = []
    for row, result in zip(rows, results["cascade"]):
        if result.hits and row_hit(result.hits[:1], row, k=1):
            cascade_rank1.append(result.hits[0].score)
    cascade_fitted = fit_tau(cascade_rank1)
    if cascade_fitted is not None:
        index.set_tau(tau_key(index.model_id, index.model_revision, "cascade"), cascade_fitted)
        fitted_by_mode["cascade"] = cascade_fitted
    fitted = fitted_by_mode.get("rerank")
    lines: list[Score] = []
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
            taken = [
                ms
                for _row_id, sample_kind, ms in samples
                if kind == "all" or sample_kind == kind
            ]
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
    if split is not None:
        hold_ids = {row["id"] for row in rows_for_split(rows, split, "holdout")}
        for mode in MODES:
            chosen = [
                (row, result)
                for row, result in zip(rows, results[mode])
                if row.get("id") in hold_ids
            ]
            if not chosen:
                continue
            taken = [ms for row_id, _kind, ms in timings.get(mode, []) if row_id in hold_ids]
            graded = [(row, result) for row, result in chosen if row.get("must_contain")]
            answerable = graded
            unanswerable = [
                (row, result) for row, result in chosen if row.get("kind") == "unanswerable"
            ]
            n = len(chosen)
            lines.append(
                Score(
                    mode=mode,
                    kind="holdout",
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
                    answerable_abstain=(
                        sum(result.reason == "no_confident_hit" for _row, result in answerable)
                        / len(answerable)
                        if answerable
                        else None
                    ),
                    unanswerable_abstain=(
                        sum(result.reason == "no_confident_hit" for _row, result in unanswerable)
                        / len(unanswerable)
                        if unanswerable
                        else None
                    ),
                )
            )
    return lines, fitted


def _mean_key(rows: list[dict[str, Any]] | None, key: str) -> float | None:
    if not rows:
        return None
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return cast(float, sum(values) / len(values))


def format_scores(lines: list[Score], fitted: float | None) -> str:
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
        if (
            line.groundedness is not None
            and line.citation_precision is not None
            and line.citation_recall is not None
        ):
            base += (
                f" ground={line.groundedness:.2f} cite_p={line.citation_precision:.2f} "
                f"cite_r={line.citation_recall:.2f}"
            )
        rendered.append(base)
    rendered.append("tau unset" if fitted is None else f"tau={fitted:.4f}")
    rendered.append(f"live={pick_live(lines)}")
    return "\n".join(rendered)


# Warm retrieve budget from §08 / suite.yaml. Eval asserts these; CI fails a regression.
RETRIEVE_P95_MS = 400.0
TTFT_P95_MS = 1200.0


def _score_map(lines: list[Score]) -> dict[tuple[str, str], Score]:
    return {(line.mode, line.kind): line for line in lines}


def _pick_gate_line(lines: list[Score], live_mode: str | None = None) -> Score | None:
    by_key = _score_map(lines)
    for mode in (live_mode, "rrf", "rerank", "cascade", "dense", "bm25"):
        if mode and (mode, "all") in by_key:
            return by_key[(mode, "all")]
    overall = [line for line in lines if line.kind == "all"]
    return overall[0] if overall else None


def check_slos(
    lines: list[Score],
    *,
    ttft_p95: float | None = None,
    suite: dict[str, Any] | None = None,
    live_mode: str | None = None,
) -> list[str]:
    """Return human-readable SLO failures. Empty list means green.

    When live_mode is set, only that arm is compared to the retrieve budget.
    """
    slo = (suite or {}).get("slo") or {}
    retrieve_budget = float(slo.get("retrieve_p95_ms", RETRIEVE_P95_MS))
    ttft_budget = float(slo.get("ttft_p95_ms", TTFT_P95_MS))
    failures = []
    for line in lines:
        if live_mode is not None and line.mode != live_mode:
            continue
        if live_mode is None and line.mode not in ("cascade", "rrf", "rerank"):
            continue
        if line.kind == "all" and line.p95 > retrieve_budget:
            failures.append(
                f"retrieve p95 {line.p95:.0f}ms exceeds {retrieve_budget:.0f}ms ({line.mode})"
            )
    if ttft_p95 is not None and ttft_p95 > ttft_budget:
        failures.append(f"ttft p95 {ttft_p95:.0f}ms exceeds {ttft_budget:.0f}ms")
    return failures


def check_gates(
    suite: dict[str, Any],
    lines: list[Score],
    *,
    answers: dict[str, str] | None = None,
    rows: list[dict[str, Any]] | None = None,
    live_mode: str | None = None,
    split: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
) -> list[str]:
    """Evaluate suite.yaml floors. Returns failure strings (empty = pass)."""
    failures: list[str] = []
    gates = suite.get("gates") or {}
    line = _pick_gate_line(lines, live_mode)
    by_key = _score_map(lines)

    def _metric(name: str, metric_line: Score | None) -> float | None:
        if metric_line is None:
            return None
        mapping = {
            "recall_at_5": metric_line.recall,
            "mrr": metric_line.mrr,
            "groundedness": metric_line.groundedness,
            "citation_precision": metric_line.citation_precision,
            "unanswerable_abstention": metric_line.unanswerable_abstain,
            "answerable_abstention": metric_line.answerable_abstain,
        }
        return mapping.get(name)

    for name, rule in gates.items():
        if name in ("conflict_disclosure", "injection_resisted"):
            continue
        if not isinstance(rule, dict):
            continue
        scope = rule.get("scope")
        metric_line = line
        if scope == "holdout":
            metric_line = by_key.get((line.mode, "holdout")) if line is not None else None
            if metric_line is None:
                failures.append(f"{name}: holdout scores missing")
                continue
        value = _metric(name, metric_line)
        if answers is not None and name in ("unanswerable_abstention", "answerable_abstention"):
            scoped = rows or []
            if scope == "holdout" and split is not None:
                hold_ids = {row["id"] for row in rows_for_split(rows or [], split, "holdout")}
                scoped = [row for row in scoped if row.get("id") in hold_ids]
            from_answers = _abstain_from_answers(name, scoped, answers)
            if from_answers is not None:
                value = from_answers
        if value is None:
            # Retrieval-only runs leave answer-quality metrics unset; skip, do not pass.
            continue
        if "min" in rule and value < float(rule["min"]):
            scope = rule.get("scope")
            label = f"{name}={value:.3f} < {float(rule['min']):.3f}"
            if scope:
                label += f" (scope={scope})"
            failures.append(label)
        if "max" in rule and value > float(rule["max"]):
            failures.append(f"{name}={value:.3f} > {float(rule['max']):.3f}")

    if answers is not None:
        conflict_rule = gates.get("conflict_disclosure") or {}
        if conflict_rule:
            conflict_rows = [row for row in (rows or []) if row.get("kind") == "conflict"]
            if not conflict_rows:
                failures.append("conflict_disclosure: no conflict rows")
            else:
                passed = sum(
                    1 for row in conflict_rows if conflict_pass(answers.get(row["id"], ""), row)
                )
                rate = passed / len(conflict_rows)
                minimum = float(conflict_rule.get("min", 1.0))
                if rate < minimum:
                    failures.append(
                        f"conflict_disclosure={rate:.3f} < {minimum:.3f} "
                        f"({passed}/{len(conflict_rows)})"
                    )

        inj_rule = gates.get("injection_resisted") or {}
        if inj_rule:
            inj_rows = [row for row in (rows or []) if row.get("kind") == "injection"]
            if not inj_rows:
                failures.append("injection_resisted: no injection rows")
            else:
                passed = sum(
                    1 for row in inj_rows if injection_pass(answers.get(row["id"], ""), row)
                )
                rate = passed / len(inj_rows)
                minimum = float(inj_rule.get("min", 1.0))
                if rate < minimum:
                    failures.append(
                        f"injection_resisted={rate:.3f} < {minimum:.3f} "
                        f"({passed}/{len(inj_rows)})"
                    )

    live = line.mode if line is not None else live_mode
    failures.extend(check_slos(lines, suite=suite, live_mode=live))

    regression = suite.get("regression") or {}
    if baseline and line is not None and regression:
        base_scores = {
            (row.get("mode"), row.get("kind")): row for row in (baseline.get("scores") or [])
        }
        live = live_mode or baseline.get("live_mode") or line.mode
        base = base_scores.get((live, "all")) or base_scores.get((baseline.get("live_mode"), "all"))
        if base and "recall" in base:
            drop = float(base["recall"]) - line.recall
            max_drop = float(regression.get("max_recall_drop", 0.03))
            if drop > max_drop:
                failures.append(f"recall drop {drop:.3f} exceeds {max_drop:.3f}")
        if base and base.get("p95") is not None and line.p95:
            base_p95 = float(base["p95"])
            if base_p95 > 0:
                increase_pct = (line.p95 - base_p95) / base_p95 * 100
                max_pct = float(regression.get("max_p95_increase_pct", 15))
                if increase_pct > max_pct:
                    failures.append(
                        f"p95 increase {increase_pct:.1f}% exceeds {max_pct:.1f}%"
                    )

    return failures


def format_gate_report(failures: list[str]) -> str:
    if not failures:
        return "suite gates: PASS"
    return "suite gates: FAIL\n" + "\n".join(f"- {item}" for item in failures)
