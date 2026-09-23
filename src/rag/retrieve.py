import time

from rag.models import (
    AGREE_TOP,
    BM25_K,
    DENSE_K,
    EMBED_MODEL,
    EMBED_REVISION,
    FAST_GAP,
    MAX_PARENTS,
    QueryError,
    RERANK_K,
    RRF_K,
    SCORE_BAND,
    Hit,
    Retrieval,
    tau_key,
)
from rag.telemetry import StageTimer


def fuse(id_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    # k=60 is Cormack's constant. A hit from only one list still stays in the running.
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ids in id_lists:
        seen = set()
        rank = 0
        for id_ in ids:
            if id_ in seen:
                continue
            seen.add(id_)
            rank += 1
            if id_ not in scores:
                first_seen[id_] = len(first_seen)
                scores[id_] = 0.0
            scores[id_] += 1.0 / (k + rank)
    ranked = sorted(scores, key=lambda id_: (-scores[id_], first_seen[id_]))
    return [(id_, scores[id_]) for id_ in ranked]


def agreed_parent(ranked: list[tuple[str, float]], dense_parents: list[str], bm25_parents: list[str]) -> str | None:
    if not ranked or not dense_parents or not bm25_parents:
        return None
    winner, best = ranked[0]
    if winner not in dense_parents[:AGREE_TOP] or winner not in bm25_parents[:AGREE_TOP]:
        return None
    if len(ranked) > 1 and ranked[1][1] >= best * FAST_GAP:
        return None
    return winner


def retrieve(index, query: str, embed_query, rerank=None, doc_id: str | None = None, mode: str = "cascade") -> Retrieval:
    if not query or not query.strip():
        raise QueryError("empty query")
    timer = StageTimer()
    wall = time.perf_counter()
    result = _retrieve(index, query, embed_query, rerank, doc_id, mode, timer)
    with timer.measure("gate", candidates_in=len(result.hits)) as gate_span:
        result = _gate(result, index, mode)
        gate_span["candidates_out"] = len(result.hits)
        if result.hits:
            gate_span["top_score"] = float(result.hits[0].score)
        if result.reason:
            gate_span["reason"] = result.reason
    stages = timer.as_ms()
    stages["total"] = (time.perf_counter() - wall) * 1000.0
    result.stages_ms = stages
    return result


def _gate(result: Retrieval, index, mode: str) -> Retrieval:
    if not result.hits or result.reason:
        return result
    top = result.hits[0].score
    cutoff = index.get_tau(_stored_tau_key(index, mode))
    if cutoff is not None and top < cutoff:
        return Retrieval(hits=[], reason="no_confident_hit")
    kept = [hit for hit in result.hits if _in_band(hit.score, top)]
    kept = kept[:MAX_PARENTS]
    kept = _retain_superseded_siblings(kept, result.hits)
    if len(kept) == 1:
        kept[0].confident = True
    elif kept:
        for hit in kept:
            hit.confident = False
    return Retrieval(hits=kept)


def _retain_superseded_siblings(kept: list, all_hits: list) -> list:
    """Down-rank must not erase superseded peers — conflict detection needs them."""
    groups = {
        getattr(hit, "version_group", None)
        for hit in kept
        if getattr(hit, "version_group", None) and not getattr(hit, "superseded", False)
    }
    if not groups:
        return kept
    present = {hit.parent_id for hit in kept}
    out = list(kept)
    for hit in all_hits:
        group = getattr(hit, "version_group", None)
        if not group or group not in groups:
            continue
        if not getattr(hit, "superseded", False):
            continue
        if hit.parent_id in present:
            continue
        out.append(hit)
        present.add(hit.parent_id)
        # One superseded peer per group is enough to disclose.
        groups.discard(group)
        if not groups:
            break
    return out


def _retrieve(index, query, embed_query, rerank, doc_id, mode, timer: StageTimer) -> Retrieval:
    dense = []
    lexical = []
    if mode != "bm25":
        with timer.measure("embed", model=getattr(index, "model_id", EMBED_MODEL)):
            vector = embed_query(query)
        with timer.measure("dense") as dense_span:
            dense = index.dense_search(vector, DENSE_K, doc_id)
            dense_span["candidates_out"] = len(dense)
    if mode != "dense":
        with timer.measure("lexical") as lex_span:
            lexical = index.bm25_search(query, BM25_K, doc_id)
            lex_span["candidates_out"] = len(lexical)
    if not dense and not lexical:
        return Retrieval(hits=[])

    chunks = {}
    for chunk in list(dense) + list(lexical):
        chunks.setdefault(chunk["chunk_id"], chunk)
    dense_parents = _parent_order(dense)
    lexical_parents = _parent_order(lexical)
    with timer.measure(
        "fuse",
        candidates_in=len(dense_parents) + len(lexical_parents),
    ) as fuse_span:
        parent_ranked = fuse([dense_parents, lexical_parents])
        child_ranked = fuse(
            [[chunk["chunk_id"] for chunk in dense], [chunk["chunk_id"] for chunk in lexical]]
        )
        fuse_span["candidates_out"] = len(parent_ranked)
        if parent_ranked:
            fuse_span["top_score"] = float(parent_ranked[0][1])
    child_scores = dict(child_ranked)

    if mode == "dense":
        return _from_parent_ids(index, dense_parents, chunks, child_scores, _rank_scores(dense_parents), False)
    if mode == "bm25":
        return _from_parent_ids(index, lexical_parents, chunks, child_scores, _rank_scores(lexical_parents), False)
    if mode == "rrf":
        return _from_parent_ids(
            index,
            [parent_id for parent_id, _score in parent_ranked],
            chunks,
            child_scores,
            dict(parent_ranked),
            False,
        )

    if mode == "cascade":
        winner = agreed_parent(parent_ranked, dense_parents, lexical_parents)
        if winner:
            return _from_parent_ids(index, [winner], chunks, child_scores, dict(parent_ranked), True)

    top_ids = [chunk_id for chunk_id, _score in child_ranked][:RERANK_K]
    top_ids = [chunk_id for chunk_id in top_ids if chunk_id in chunks]
    if rerank is None:
        raise QueryError("reranker is not available")
    with timer.measure("rerank", candidates_in=len(top_ids)) as rerank_span:
        scores = list(rerank(query, [chunks[chunk_id]["embed_text"] for chunk_id in top_ids]))
        if len(scores) != len(top_ids):
            raise QueryError("reranker returned the wrong number of scores")
        order = sorted(range(len(top_ids)), key=lambda index: -scores[index])
        rerank_span["candidates_out"] = len(order)
        if scores:
            rerank_span["top_score"] = float(max(scores))
    parent_ids = []
    parent_scores = {}
    child_for = {}
    for index_ in order:
        parent_id = chunks[top_ids[index_]]["parent_id"]
        if parent_id in parent_scores:
            continue
        parent_scores[parent_id] = float(scores[index_])
        child_for[parent_id] = top_ids[index_]
        parent_ids.append(parent_id)

    if mode == "rerank":
        return _emit(index, parent_ids[:MAX_PARENTS], parent_scores, child_for, False)

    if not parent_ids:
        return Retrieval(hits=[])
    return _emit(index, parent_ids[:MAX_PARENTS], parent_scores, child_for, False)


def _stored_tau_key(index, mode: str = "cascade") -> str:
    return tau_key(
        getattr(index, "model_id", EMBED_MODEL),
        getattr(index, "model_revision", EMBED_REVISION),
        mode,
    )


def _in_band(score: float, top: float) -> bool:
    return score >= top - abs(top) * SCORE_BAND


def _parent_order(chunks: list[dict]) -> list[str]:
    ordered = []
    seen = set()
    for chunk in chunks:
        parent_id = chunk["parent_id"]
        if parent_id in seen:
            continue
        seen.add(parent_id)
        ordered.append(parent_id)
    return ordered


def _rank_scores(parent_ids: list[str]) -> dict[str, float]:
    return {parent_id: 1.0 / (index + 1) for index, parent_id in enumerate(parent_ids)}


def _from_parent_ids(index, parent_ids, chunks, child_scores, scores, confident) -> Retrieval:
    by_parent: dict[str, list] = {}
    for chunk in chunks.values():
        by_parent.setdefault(chunk["parent_id"], []).append(chunk)
    child_for = {}
    for parent_id in parent_ids[:MAX_PARENTS]:
        child_for[parent_id] = _best_child(parent_id, by_parent.get(parent_id, []), child_scores)
    return _emit(index, parent_ids[:MAX_PARENTS], scores, child_for, confident)


def _best_child(parent_id: str, parent_chunks: list, child_scores: dict) -> str:
    del parent_id
    best_id = ""
    best_score = None
    for chunk in parent_chunks:
        score = child_scores.get(chunk["chunk_id"], 0.0)
        if best_score is None or score > best_score:
            best_id = chunk["chunk_id"]
            best_score = score
    return best_id


def _emit(index, parent_ids, scores, child_for, confident: bool) -> Retrieval:
    from rag.versions import downrank_superseded

    records = index.get_parents(parent_ids)
    hits = []
    for parent_id in parent_ids:
        record = records.get(parent_id)
        if record is None or not child_for.get(parent_id):
            continue
        status = record.get("status") or "current"
        hits.append(
            Hit(
                parent_id=parent_id,
                parent_text=record["text"],
                heading_path=record.get("heading_path") or "",
                source_path=record.get("source_path") or "",
                file_sha256=record.get("file_sha256") or "",
                page_start=int(record.get("page_start") or 0),
                page_end=int(record.get("page_end") or 0),
                start_char=int(record.get("start_char") or 0),
                end_char=int(record.get("end_char") or 0),
                child_id=child_for[parent_id],
                score=float(scores[parent_id]),
                confident=confident,
                derived=bool(record.get("derived")),
                superseded=bool(record.get("superseded")) or status == "superseded",
                version_group=record.get("version_group"),
                status=status,
                superseded_by=record.get("superseded_by"),
                supersedes=record.get("supersedes"),
                review_date=record.get("review_date"),
                title=record.get("title"),
            )
        )
    downrank_superseded(hits)
    return Retrieval(hits=hits)
