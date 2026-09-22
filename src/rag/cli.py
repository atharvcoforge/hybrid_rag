import argparse
import json
import sys
from pathlib import Path

from rag.evaluate import evaluate, format_scores, load_rows, load_split
from rag.models import IngestError, QueryError
from rag.pipeline import ingest, query


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_cmd = sub.add_parser("ingest", help="index files")
    ingest_cmd.add_argument("path")
    ingest_cmd.add_argument("--index", required=True)

    query_cmd = sub.add_parser("query", help="search the index")
    query_cmd.add_argument("text")
    query_cmd.add_argument("--index", required=True)
    query_cmd.add_argument("--doc", default=None)

    eval_cmd = sub.add_parser("eval", help="score a golden set")
    eval_cmd.add_argument("--index", required=True)
    eval_cmd.add_argument("--golden", required=True)

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
            print(_run_eval(args.index, args.golden))
    except (IngestError, QueryError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _run_eval(index_dir, golden) -> str:
    from rag.embed import encode_query, rerank_scores
    from rag.models import EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION
    from rag.retrieve import retrieve
    from rag.store import Index

    rows = load_rows(golden)
    index = Index(index_dir, EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION)
    index.open()
    try:
        def ask(mode, row):
            return retrieve(index, row["q"], encode_query, rerank=rerank_scores, mode=mode)

        lines, fitted = evaluate(index, rows, ask)
    finally:
        index.close()
    return format_scores(lines, fitted)


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
