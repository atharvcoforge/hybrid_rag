import argparse
import sys

from rag.evaluate import evaluate, format_scores, load_rows
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


def _format_hit(hit) -> str:
    pages = f"p.{hit.page_start}-{hit.page_end}" if hit.page_start else "p.-"
    flag = "confident" if hit.confident else "uncertain"
    head = (
        f"{hit.score:.4f}  {flag}  {hit.source_path}  {hit.heading_path}  "
        f"{pages}  chars {hit.start_char}-{hit.end_char}  {hit.file_sha256[:12]}"
    )
    return head + "\n" + hit.parent_text
