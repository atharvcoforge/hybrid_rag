import argparse
import json
import sys
from pathlib import Path

from rag.evaluate import (
    DEFAULT_SUITE,
    assert_kind_coverage,
    evaluate,
    format_gate_report,
    format_scores,
    load_rows,
    load_split,
    load_suite,
    verify_corpus,
)
from rag.models import IngestError, QueryError
from rag.pipeline import ingest, query


def main(argv=None) -> int:
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

    args = parser.parse_args(argv)
    from rag.telemetry import configure_logging

    configure_logging(args.log_format)
    try:
        if args.cmd == "ingest":
            for item in ingest(args.path, args.index):
                print(f"{item.status}  {item.doc_id}  {item.chunks}")
        elif args.cmd == "query":
            result = query(args.text, args.index, doc_id=args.doc)
            if result.reason:
                print(result.reason)
            elif not result.hits:
                print("no hits")
            else:
                print("\n\n".join(_format_hit(hit) for hit in result.hits))
        elif args.cmd == "purge":
            for item in ingest(args.path, args.index):
                if item.status == "purged":
                    print(f"purged  {item.doc_id}")
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


def _run_eval(index_dir, golden, suite_path) -> tuple[str, bool]:
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
    try:
        def ask(mode, row):
            return retrieve(index, row["q"], encode_query, rerank=rerank_scores, mode=mode)

        lines, fitted = evaluate(index, rows, ask)
    finally:
        index.close()
    body = format_scores(lines, fitted)
    if suite is None:
        return body, True
    split_path = suite.get("split") or "evals/split.json"
    split = load_split(split_path) if Path(split_path).exists() else None
    baseline = None
    baseline_path = (suite.get("regression") or {}).get("baseline")
    if baseline_path and Path(baseline_path).exists():
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    gate_failures = check_gates(
        suite,
        lines,
        rows=rows,
        live_mode=pick_live(lines),
        split=split,
        baseline=baseline,
        answers=None,  # answer gates need generation; Part F fills them
    )
    return body + "\n" + format_gate_report(gate_failures), not gate_failures


def _run_calibrate(index_dir, golden, split_path, out_path) -> str:
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
        def ask(mode, row):
            return retrieve(index, row["q"], encode_query, rerank=rerank_scores, mode=mode)

        report = calibrate(index, rows, ask, split=split)
    finally:
        index.close()
    write_report(out_path, report)
    return json.dumps(report, indent=2)


def _format_hit(hit) -> str:
    pages = f"p.{hit.page_start}-{hit.page_end}" if hit.page_start else "p.-"
    flag = "confident" if hit.confident else "uncertain"
    head = (
        f"{hit.score:.4f}  {flag}  {hit.source_path}  {hit.heading_path}  "
        f"{pages}  chars {hit.start_char}-{hit.end_char}  {hit.file_sha256[:12]}"
    )
    return head + "\n" + hit.parent_text
