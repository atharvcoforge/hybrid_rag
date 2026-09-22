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
from rag.normalize import normalize_text
from rag.pipeline import ingest, query
from rag.telemetry import log_event, new_request_id

_answers = {}

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


TITLES = {
    "Carbon_New_2040.pdf": "Carbon Reduction Plan",
    "Envi_2040-1.pdf": "Environmental Sustainability Policy",
    "Environmental_Sustainability_Policy_2025.pdf": "Environmental Sustainability Policy (2025)",
    "Water-Management-Policy.pdf": "Water Management Policy",
}


def hit_dict(hit, cite: int) -> dict:
    source = hit.source_path
    return {
        "parent_id": hit.parent_id,
        "text": hit.parent_text,
        "heading": hit.heading_path,
        "source": source,
        "title": TITLES.get(Path(source).name, Path(source).name),
        "cite": cite,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "score": hit.score,
        "confident": hit.confident,
    }


def iter_query(text, search, write, mode, cache=None):
    if cache is None:
        cache = _answers
    cleaned = normalize_text(text or "")
    request_id = new_request_id()
    if not cleaned:
        yield ("error", {"message": "empty query", "request_id": request_id})
        return
    key = cleaned.casefold()
    saved = cache.get(key)
    if saved is not None:
        yield ("meta", {**saved["meta"], "retrieve_ms": 0, "cached": True, "request_id": request_id})
        if saved["answer"]:
            yield ("token", {"t": saved["answer"]})
        yield ("done", {"first_token_ms": 0, "answer": saved["answer"], "cached": True, "request_id": request_id})
        return
    started = time.perf_counter()
    result = search(cleaned, mode)
    retrieve_ms = round((time.perf_counter() - started) * 1000)
    stages = dict(result.stages_ms or {})
    meta = {
        "hits": [hit_dict(hit, index) for index, hit in enumerate(result.hits, start=1)],
        "reason": result.reason,
        "mode": mode,
        "retrieve_ms": retrieve_ms,
        "stages_ms": stages,
        "cached": False,
        "request_id": request_id,
    }
    log_event(
        "retrieve",
        request_id=request_id,
        mode=mode,
        retrieve_ms=retrieve_ms,
        hits=len(result.hits),
        reason=result.reason or "",
        **{f"{name}_ms": value for name, value in stages.items()},
    )
    yield ("meta", meta)
    if result.reason == "no_confident_hit" or not result.hits:
        cache[key] = {"meta": meta, "answer": ""}
        yield ("done", {"first_token_ms": None, "answer": "", "request_id": request_id})
        return
    first = None
    write_started = time.perf_counter()
    parts = []
    try:
        for piece in write(cleaned, result.hits):
            if not piece:
                continue
            if first is None:
                first = round((time.perf_counter() - write_started) * 1000)
            parts.append(piece)
            yield ("token", {"t": piece})
    except (OSError, urllib.error.URLError, TimeoutError):
        yield ("error", {"message": "writer unavailable", "request_id": request_id})
        return
    answer = "".join(parts)
    cache[key] = {"meta": meta, "answer": answer}
    yield ("done", {"first_token_ms": first, "answer": answer, "request_id": request_id})


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
                "p95": line.p95,
                "p99": line.p99,
                "n": line.n,
                "groundedness": line.groundedness,
                "citation_precision": line.citation_precision,
                "citation_recall": line.citation_recall,
                "answerable_abstain": line.answerable_abstain,
                "unanswerable_abstain": line.unanswerable_abstain,
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
