import json
import os
import time
import urllib.error
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.evaluate import evaluate, load_rows, pick_live, row_hit
from rag.generate import complete, stream_answer, writer_up
from rag.models import EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION
from rag.pipeline import ingest, query

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryBody(BaseModel):
    q: str


class IngestBody(BaseModel):
    path: str | None = None


def index_dir() -> Path:
    return Path(os.environ.get("INDEX_DIR", "index"))


def eval_path() -> Path:
    return Path(os.environ.get("EVAL_OUT", "evals/latest.json"))


def golden_path() -> Path:
    return Path(os.environ.get("GOLDEN", "evals/policy.jsonl"))


def live_mode() -> str:
    override = os.environ.get("LIVE_MODE")
    if override:
        return override
    path = eval_path()
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return saved.get("live_mode") or "cascade"
    return "cascade"


def hit_dict(hit) -> dict:
    return {
        "parent_id": hit.parent_id,
        "text": hit.parent_text,
        "heading": hit.heading_path,
        "source": hit.source_path,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "score": hit.score,
        "confident": hit.confident,
    }


def iter_query(text, search, write, mode):
    if not text or not text.strip():
        yield ("error", {"message": "empty query"})
        return
    started = time.perf_counter()
    result = search(text, mode)
    retrieve_ms = round((time.perf_counter() - started) * 1000)
    yield (
        "meta",
        {
            "hits": [hit_dict(hit) for hit in result.hits],
            "reason": result.reason,
            "mode": mode,
            "retrieve_ms": retrieve_ms,
        },
    )
    if result.reason == "no_confident_hit" or not result.hits:
        yield ("done", {"first_token_ms": None, "answer": ""})
        return
    first = None
    write_started = time.perf_counter()
    parts = []
    try:
        for piece in write(text, result.hits):
            if not piece:
                continue
            if first is None:
                first = round((time.perf_counter() - write_started) * 1000)
            parts.append(piece)
            yield ("token", {"t": piece})
    except (OSError, urllib.error.URLError, TimeoutError):
        yield ("error", {"message": "writer unavailable"})
        return
    yield ("done", {"first_token_ms": first, "answer": "".join(parts)})


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def span_rate(rows, search, finish, mode) -> tuple[float | None, int]:
    graded = [row for row in rows if row.get("expect")]
    if not graded:
        return None, 0
    got = 0
    for row in graded:
        result = search(row["q"], mode)
        if result.reason == "no_confident_hit" or not result.hits:
            continue
        if not row_hit(result.hits, row):
            continue
        answer = finish(row["q"], result.hits)
        if row["expect"].lower() in answer.lower():
            got += 1
    return got / len(graded), len(graded)


def build_report(index_dir_path, golden, search, finish=None) -> dict:
    from rag.embed import encode_query, rerank_scores
    from rag.retrieve import retrieve
    from rag.store import Index

    rows = load_rows(golden)
    index = Index(index_dir_path, EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION)
    index.open()
    try:
        def ask_retrieve(mode, row):
            return retrieve(index, row["q"], encode_query, rerank=rerank_scores, mode=mode)

        lines, fitted = evaluate(index, rows, ask_retrieve)
    finally:
        index.close()
    mode = pick_live(lines)
    span = None
    span_n = 0
    if finish is not None and writer_up():
        span, span_n = span_rate(rows, search, finish, mode)
    report = {
        "tau": fitted,
        "live_mode": mode,
        "scores": [
            {
                "mode": line.mode,
                "kind": line.kind,
                "recall": line.recall,
                "mrr": line.mrr,
                "abstain": line.abstain,
                "p50": line.p50,
                "n": line.n,
            }
            for line in lines
        ],
        "span_hit": span,
        "span_n": span_n,
    }
    report["verdict"] = _verdict(report)
    return report


def _verdict(report: dict) -> str:
    scores = {(row["mode"], row["kind"]): row for row in report["scores"]}
    live = report["live_mode"]
    chosen = scores[(live, "all")]
    rerank = scores[("rerank", "all")]
    parts = [
        f"Live retrieval is {live}: recall@5 {chosen['recall']:.2f}, MRR {chosen['mrr']:.2f}, abstain {chosen['abstain']:.2f}, retrieve p50 {chosen['p50']:.0f} ms.",
        f"Full rerank recall@5 is {rerank['recall']:.2f} at p50 {rerank['p50']:.0f} ms.",
        f"{live} is the fastest mode within 0.05 recall of the best score on this set.",
    ]
    span = report["span_hit"]
    if span is None:
        parts.append("The writer was not scored.")
    elif span < 0.5:
        parts.append(f"Answer span-hit is {span:.2f} on {report['span_n']} rows. That is weak for the 3B model; the next step is the 7B Q4 of the same family.")
    else:
        parts.append(f"Answer span-hit is {span:.2f} on {report['span_n']} rows, so the 3B model stays.")
    parts.append("Still not a production service: one machine, no auth, and the golden set was written by hand against three policies.")
    return " ".join(parts)


def _search(text, mode):
    return query(text, index_dir(), mode=mode)


@app.get("/api/health")
def health():
    return {"index": (index_dir() / "side.sqlite").exists(), "writer": writer_up()}


@app.post("/api/query")
def ask(body: QueryBody):
    mode = live_mode()

    def gen():
        for event, data in iter_query(body.q, _search, stream_answer, mode):
            yield sse(event, data)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/ingest")
def ingest_route(body: IngestBody | None = None):
    path = (body.path if body and body.path else None) or os.environ.get("CORPUS", "documents")
    items = ingest(path, index_dir())
    return {"items": [{"doc_id": item.doc_id, "status": item.status, "chunks": item.chunks} for item in items]}


@app.get("/api/eval")
def eval_get():
    path = eval_path()
    if not path.exists():
        return {"scores": [], "live_mode": "cascade", "verdict": ""}
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/eval")
def eval_post():
    report = build_report(index_dir(), golden_path(), _search, complete)
    path = eval_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
