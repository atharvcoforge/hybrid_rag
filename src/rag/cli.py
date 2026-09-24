import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rag.evaluate import (
    DEFAULT_SUITE,
    Score,
    assert_kind_coverage,
    evaluate,
    format_gate_report,
    format_scores,
    load_rows,
    load_split,
    load_suite,
    verify_corpus,
)
from rag.gates import GateResult
from rag.models import Hit, IngestError, QueryError, Retrieval
from rag.pipeline import ingest, query


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    parser.add_argument(
        "--log-format",
        choices=("json", "console"),
        default=None,
        help="structured json (default) or human console lines",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_cmd = sub.add_parser("ingest", help="index files")
    ingest_cmd.add_argument("path")
    ingest_cmd.add_argument("--index", required=True)

    query_cmd = sub.add_parser("query", help="search the index")
    query_cmd.add_argument("text")
    query_cmd.add_argument("--index", required=True)
    query_cmd.add_argument("--doc", default=None)
    query_cmd.add_argument(
        "--mode",
        default="cascade",
        choices=("dense", "bm25", "rrf", "rerank", "cascade"),
    )

    ask_cmd = sub.add_parser("ask", help="retrieve, answer, and cite")
    ask_cmd.add_argument("text")
    ask_cmd.add_argument("--index", required=True)
    ask_cmd.add_argument(
        "--mode",
        default=None,
        choices=("dense", "bm25", "rrf", "rerank", "cascade"),
    )

    eval_cmd = sub.add_parser("eval", help="score a golden set against suite.yaml gates")
    eval_cmd.add_argument("--index", required=True)
    eval_cmd.add_argument("--golden", default=None, help="JSONL rows (default: suite.rows)")
    eval_cmd.add_argument("--suite", default=DEFAULT_SUITE, help="evals/suite.yaml")

    calibrate_cmd = sub.add_parser("calibrate", help="fit confidence thresholds")
    calibrate_cmd.add_argument("--index", required=True)
    calibrate_cmd.add_argument("--golden", required=True)
    calibrate_cmd.add_argument("--split", default="evals/split.json")
    calibrate_cmd.add_argument("--out", default="evals/calibration.json")

    purge_cmd = sub.add_parser("purge", help="drop docs missing from a folder")
    purge_cmd.add_argument("path")
    purge_cmd.add_argument("--index", required=True)
    purge_cmd.add_argument("--missing", action="store_true", required=True)

    verify_cmd = sub.add_parser("verify", help="check index integrity")
    verify_cmd.add_argument("--index", required=True)

    args = parser.parse_args(argv)
    from rag.telemetry import configure_logging

    configure_logging(args.log_format)
    try:
        if args.cmd == "ingest":
            for item in ingest(args.path, args.index):
                print(f"{item.status}  {item.doc_id}  {item.chunks}")
        elif args.cmd == "query":
            result = query(args.text, args.index, doc_id=args.doc, mode=args.mode)
            if result.reason:
                print(result.reason)
            elif not result.hits:
                print("no hits")
            else:
                print("\n\n".join(_format_hit(hit) for hit in result.hits))
        elif args.cmd == "ask":
            print(_run_ask(args.text, args.index, args.mode))
        elif args.cmd == "purge":
            for item in ingest(args.path, args.index):
                if item.status == "purged":
                    print(f"purged  {item.doc_id}")
        elif args.cmd == "verify":
            problems = _run_verify(args.index)
            if problems:
                print("\n".join(problems), file=sys.stderr)
                return 1
            print("integrity clean")
        elif args.cmd == "calibrate":
            print(_run_calibrate(args.index, args.golden, args.split, args.out))
        else:
            text, ok = _run_eval(args.index, args.golden, args.suite)
            print(text)
            return 0 if ok else 1
    except (IngestError, QueryError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _run_verify(index_dir: str) -> list[str]:
    from rag.models import EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION
    from rag.store import Index

    index = Index(index_dir, EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION)
    index.open()
    try:
        return index.integrity_problems()
    finally:
        index.close()


def _run_eval(index_dir: str, golden: str | None, suite_path: str | None) -> tuple[str, bool]:
    from rag.embed import encode_query, rerank_scores
    from rag.evaluate import check_gates, pick_live
    from rag.models import EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION
    from rag.retrieve import retrieve
    from rag.store import Index

    suite = load_suite(suite_path) if suite_path and Path(suite_path).exists() else None
    if suite is not None:
        pin_failures = verify_corpus(suite)
        kind_path = golden or suite.get("rows") or "evals/policy.jsonl"
        rows = load_rows(kind_path)
        kind_failures = assert_kind_coverage(rows)
        if pin_failures or kind_failures:
            parts = []
            if pin_failures:
                parts.append("corpus pins:\n" + "\n".join(f"- {item}" for item in pin_failures))
            if kind_failures:
                parts.append("kinds:\n" + "\n".join(f"- {item}" for item in kind_failures))
            return "\n".join(parts), False
        golden = kind_path
    else:
        if not golden:
            raise QueryError("--golden is required when suite.yaml is absent")
        rows = load_rows(golden)

    index = Index(index_dir, EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION)
    index.open()
    split = None
    if suite is not None:
        split_path = suite.get("split") or "evals/split.json"
        split = load_split(split_path) if Path(split_path).exists() else None
    slo_limit = None
    if suite is not None:
        slo_limit = (suite.get("slo") or {}).get("retrieve_p95_ms")
    try:
        def ask(mode: str, row: dict[str, Any]) -> Retrieval:
            return retrieve(index, row["q"], encode_query, rerank=rerank_scores, mode=mode)

        lines, fitted = evaluate(index, rows, ask, split=split)
        live = pick_live(lines, p95_limit=slo_limit)
        index._set_meta("live_mode", live)
        answers = None
        if suite is not None:
            answers, quality = _answers_for_rows(rows, ask, live)
            _stamp_quality(lines, live, quality)
    finally:
        index.close()
    body = format_scores(lines, fitted, p95_limit=slo_limit)
    if suite is None:
        return body, True
    baseline = None
    baseline_path = (suite.get("regression") or {}).get("baseline")
    if baseline_path and Path(baseline_path).exists():
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    gate_failures = check_gates(
        suite,
        lines,
        rows=rows,
        live_mode=live,
        split=split,
        baseline=baseline,
        answers=answers,
    )
    if answers is None:
        gate_failures.append(
            "generator down: answer abstention, conflict_disclosure, and injection_resisted were not scored"
        )
    return body + "\n" + format_gate_report(gate_failures), not gate_failures


def _answers_for_rows(
    rows: list[dict[str, Any]],
    ask: Callable[[str, dict[str, Any]], Retrieval],
    live: str,
) -> tuple[dict[str, str] | None, dict[str, float | None] | None]:
    from rag.gates import check_all
    from rag.generate import complete, writer_up

    if not writer_up():
        return None, None
    answers: dict[str, str] = {}
    grounded: list[float] = []
    cited: list[float] = []
    for row in rows:
        result = ask(live, row)
        text, gate = _gated_text(row.get("q") or "", result, complete, check_all)
        answers[row["id"]] = text
        if gate is None:
            continue
        if gate.reason == "unsupported_figure":
            grounded.append(0.0)
            cited.append(0.0)
        elif gate.ok and gate.reason != "declined":
            grounded.append(1.0 if gate.groundedness is None else float(gate.groundedness))
            cited.append(1.0 if gate.citation_precision is None else float(gate.citation_precision))
    quality = {
        "groundedness": sum(grounded) / len(grounded) if grounded else None,
        "citation_precision": sum(cited) / len(cited) if cited else None,
    }
    return answers, quality


def _stamp_quality(
    lines: list[Score], live: str, quality: dict[str, float | None] | None
) -> None:
    if not quality:
        return
    for line in lines:
        if line.mode == live and line.kind == "all":
            line.groundedness = quality.get("groundedness")
            line.citation_precision = quality.get("citation_precision")
            line.citation_recall = quality.get("citation_precision")


def _gated_text(
    query: str,
    result: Retrieval,
    complete: Callable[[str, Sequence[Hit]], str],
    check_all: Callable[..., GateResult],
) -> tuple[str, GateResult | None]:
    if getattr(result, "reason", "") == "no_confident_hit" or not result.hits:
        return "", None
    text = complete(query, result.hits)
    gate = check_all(text, result.hits, query=query)
    if gate.answer:
        text = gate.answer
    if not gate.ok or gate.state == "withheld":
        return "The documents do not say.", gate
    return text, gate


def _run_calibrate(index_dir: str, golden: str, split_path: str, out_path: str) -> str:
    from rag.calibrate import calibrate, write_report
    from rag.embed import encode_query, rerank_scores
    from rag.models import EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION
    from rag.retrieve import retrieve
    from rag.store import Index

    rows = load_rows(golden)
    split = load_split(split_path) if Path(split_path).exists() else None
    index = Index(index_dir, EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION)
    index.open()
    try:
        def ask(mode: str, row: dict[str, Any]) -> Retrieval:
            return retrieve(index, row["q"], encode_query, rerank=rerank_scores, mode=mode)

        report = calibrate(index, rows, ask, split=split)
    finally:
        index.close()
    write_report(out_path, report)
    return json.dumps(report, indent=2)


def _live_mode(index_dir: str) -> str:
    from rag.models import EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION
    from rag.store import Index

    index = Index(index_dir, EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION)
    index.open()
    try:
        saved = index._meta("live_mode")
    finally:
        index.close()
    return saved or "rrf"


def _run_ask(text: str, index_dir: str, mode: str | None) -> str:
    from rag.gates import check_all
    from rag.generate import complete

    chosen = mode or _live_mode(index_dir)
    result = query(text, index_dir, mode=chosen)
    if result.reason:
        return result.reason
    if not result.hits:
        return "no hits"
    answer, _gate = _gated_text(text, result, complete, check_all)
    lines = [answer, ""]
    for number, hit in enumerate(result.hits, start=1):
        status = "SUPERSEDED" if hit.superseded or hit.status == "superseded" else "CURRENT"
        pages = f"p.{hit.page_start}-{hit.page_end}" if hit.page_start else "p.-"
        lines.append(f"[{number}] {status}  {hit.source_path}  {hit.heading_path}  {pages}")
    return "\n".join(lines)


def _format_hit(hit: Hit) -> str:
    pages = f"p.{hit.page_start}-{hit.page_end}" if hit.page_start else "p.-"
    flag = "confident" if hit.confident else "uncertain"
    head = (
        f"{hit.score:.4f}  {flag}  {hit.source_path}  {hit.heading_path}  "
        f"{pages}  chars {hit.start_char}-{hit.end_char}  {hit.file_sha256[:12]}"
    )
    return head + "\n" + hit.parent_text
