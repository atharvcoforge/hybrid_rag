import math
import time
from collections.abc import Callable
from typing import Any, cast

from rag.models import (
    AGREE_TOP,
    BM25_K,
    DENSE_K,
    EMBED_MODEL,
    EMBED_REVISION,
    FAST_GAP,
    MAX_PARENTS,
    RERANK_K,
    RRF_K,
    SCORE_BAND,
    Hit,
    QueryError,
    Retrieval,
    tau_key,
)
from rag.store import Index
from rag.telemetry import StageTimer
from rag.versions import SUPERSEDED_SCORE_FACTOR, title_match_boost


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


def retrieve(
    index: Index,
    query: str,
    embed_query: Callable[[str], list[float]],
    rerank: Callable[[str, list[str]], list[float]] | None = None,
    doc_id: str | None = None,
    mode: str = "cascade",
) -> Retrieval:
    if not query or not query.strip():
        raise QueryError("empty query")
    timer = StageTimer()
    wall = time.perf_counter()
    result = _retrieve(index, query, embed_query, rerank, doc_id, mode, timer)
    with timer.measure("gate", candidates_in=len(result.hits)) as gate_span:
        result = _gate(result, index, mode, query)
        gate_span["candidates_out"] = len(result.hits)
        if result.hits:
            gate_span["top_score"] = float(result.hits[0].score)
        if result.reason:
            gate_span["reason"] = result.reason
    stages = timer.as_ms()
    stages["total"] = (time.perf_counter() - wall) * 1000.0
    result.stages_ms = stages
    return result


def _gate(result: Retrieval, index: Index, mode: str, query: str = "") -> Retrieval:
    if not result.hits or result.reason:
        return result
    # Tau judges the best retriever score. Title ordering may place a
    # lower-scored passage first; that must not turn a confident list into an abstention.
    top = max(hit.score for hit in result.hits)
    cutoff = index.get_tau(_stored_tau_key(index, mode))
    if cutoff is not None and top < cutoff:
        return Retrieval(hits=[], reason="no_confident_hit")
    # The band marks confidence. It does not drop hits: rank-2 reciprocal
    # scores sit at half of rank 1, so a cut here used to turn recall@5 into precision@1.
    kept = list(result.hits)[:MAX_PARENTS]
    kept = _retain_superseded_siblings(kept, result.hits)
    kept = _append_fact_siblings(index, query, kept)
    if len(kept) == 1:
        kept[0].confident = True
    else:
        for hit in kept:
            hit.confident = mode in ("rerank", "cascade") and _in_band(hit.score, top)
    return Retrieval(hits=kept)


def _retain_superseded_siblings(kept: list[Hit], all_hits: list[Hit]) -> list[Hit]:
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


def _append_fact_siblings(index: Index, query: str, hits: list[Hit]) -> list[Hit]:
    """Keep the paired version of a matched fact even when it lost the top-k cut."""
    from rag.versions import aligned_peer_records

    records = aligned_peer_records(index, hits)
    if not records:
        return hits
    out = list(hits)
    present = {hit.parent_id for hit in out}
    added: set[str] = set()
    for record in records:
        parent_id = record.get("parent_id")
        if not parent_id or parent_id in present:
            continue
        status = record.get("status") or "current"
        out.append(
            Hit(
                parent_id=parent_id,
                parent_text=record.get("text") or "",
                heading_path=record.get("heading_path") or "",
                source_path=record.get("source_path") or "",
                file_sha256=record.get("file_sha256") or "",
                page_start=int(record.get("page_start") or 0),
                page_end=int(record.get("page_end") or 0),
                start_char=int(record.get("start_char") or 0),
                end_char=int(record.get("end_char") or 0),
                child_id=record.get("child_id") or parent_id,
                score=0.0,
                confident=False,
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
        present.add(parent_id)
        added.add(parent_id)
    while len(out) > MAX_PARENTS:
        removable = [hit for hit in out if hit.parent_id not in added]
        if not removable:
            break
        worst = min(removable, key=lambda hit: float(hit.score))
        out.remove(worst)
    return out


def _retrieve(
    index: Index,
    query: str,
    embed_query: Callable[[str], list[float]],
    rerank: Callable[[str, list[str]], list[float]] | None,
    doc_id: str | None,
    mode: str,
    timer: StageTimer,
) -> Retrieval:
    dense: list[dict[str, Any]] = []
    lexical: list[dict[str, Any]] = []
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

    chunks: dict[str, dict[str, Any]] = {}
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
        order, scores = _parent_scores_from_chunks(dense, "dense_score")
        order = _order_current_first(index, order, scores, query)
        return _from_parent_ids(index, order, chunks, child_scores, scores, False, query)
    if mode == "bm25":
        order, scores = _parent_scores_from_chunks(lexical, "bm25_score")
        order = _order_current_first(index, order, scores, query)
        return _from_parent_ids(index, order, chunks, child_scores, scores, False, query)
    if mode == "rrf":
        score_map = dict(parent_ranked)
        order = _order_current_first(
            index, [parent_id for parent_id, _score in parent_ranked], score_map, query
        )
        return _from_parent_ids(
            index,
            order,
            chunks,
            child_scores,
            score_map,
            False,
            query,
        )

    if mode == "cascade":
        winner = agreed_parent(parent_ranked, dense_parents, lexical_parents)
        if winner:
            return _from_parent_ids(
                index, [winner], chunks, child_scores, dict(parent_ranked), True, query
            )

    ranked_parents = [parent_id for parent_id, _score in parent_ranked][:RERANK_K]
    records = index.get_parents(ranked_parents)
    passages: list[str] = []
    top_ids: list[str] = []
    for parent_id in ranked_parents:
        record = records.get(parent_id)
        if record is None:
            continue
        heading = record.get("heading_path") or ""
        body = record.get("text") or ""
        passages.append(f"{heading}\n{body}" if heading else body)
        top_ids.append(parent_id)
    if not top_ids:
        return Retrieval(hits=[])
    if rerank is None:
        raise QueryError("reranker is not available")
    with timer.measure("rerank", candidates_in=len(top_ids)) as rerank_span:
        raw_scores = list(rerank(query, passages))
        if len(raw_scores) != len(top_ids):
            raise QueryError("reranker returned the wrong number of scores")
        rerank_scores = [_sigmoid(float(score)) for score in raw_scores]
        rerank_order = sorted(range(len(top_ids)), key=lambda index: -rerank_scores[index])
        rerank_span["candidates_out"] = len(rerank_order)
        rerank_span["top_score"] = float(max(rerank_scores))
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks.values():
        by_parent.setdefault(cast(str, chunk["parent_id"]), []).append(chunk)
    parent_ids: list[str] = []
    parent_scores: dict[str, float] = {}
    child_for: dict[str, str] = {}
    for index_ in rerank_order:
        parent_id = top_ids[index_]
        parent_scores[parent_id] = rerank_scores[index_]
        child_for[parent_id] = _best_child(parent_id, by_parent.get(parent_id, []), child_scores)
        parent_ids.append(parent_id)

    if mode == "rerank":
        parent_ids = _order_current_first(index, parent_ids, parent_scores, query)
        return _emit(index, parent_ids[:MAX_PARENTS], parent_scores, child_for, False, query, focus=False)

    parent_ids = _order_current_first(index, parent_ids, parent_scores, query)
    return _emit(index, parent_ids[:MAX_PARENTS], parent_scores, child_for, False, query, focus=False)


def _stored_tau_key(index: Index, mode: str = "cascade") -> str:
    return tau_key(
        cast(str, getattr(index, "model_id", EMBED_MODEL)),
        cast(str, getattr(index, "model_revision", EMBED_REVISION)),
        mode,
    )


def _sigmoid(value: float) -> float:
    if value >= 20:
        return 1.0
    if value <= -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _in_band(score: float, top: float) -> bool:
    return score >= top - abs(top) * SCORE_BAND


def _parent_scores_from_chunks(
    chunks: list[dict[str, Any]], key: str
) -> tuple[list[str], dict[str, float]]:
    """Parent order by the best child score. Rank-reciprocal when the search did not send one."""
    if not chunks or key not in chunks[0]:
        order = _parent_order(chunks)
        return order, _rank_scores(order)
    scores: dict[str, float] = {}
    first: dict[str, int] = {}
    for chunk in chunks:
        parent_id = cast(str, chunk["parent_id"])
        if parent_id not in first:
            first[parent_id] = len(first)
        score = float(chunk[key])
        if parent_id not in scores or score > scores[parent_id]:
            scores[parent_id] = score
    order = sorted(scores, key=lambda parent_id: (-scores[parent_id], first[parent_id]))
    return order, scores


def _order_current_first(
    index: Index, parent_ids: list[str], scores: dict[str, float], query: str = ""
) -> list[str]:
    """Apply the superseded penalty before the top-k cut, so an old copy cannot take the slot."""
    if len(parent_ids) < 2:
        return list(parent_ids)
    from rag.versions import query_names_document

    records = index.get_parents(parent_ids)
    origin = {parent_id: position for position, parent_id in enumerate(parent_ids)}

    def sort_key(parent_id: str) -> tuple[float, int]:
        record = records.get(parent_id) or {}
        score = float(scores.get(parent_id, 0.0))
        status = record.get("status") or ""
        named = query_names_document(query, record.get("source_path") or "", record.get("review_date"))
        if (status == "superseded" or record.get("superseded")) and not named:
            score *= SUPERSEDED_SCORE_FACTOR
        score *= title_match_boost(query, record.get("title") or "", record.get("source_path") or "")
        return (-score, origin[parent_id])

    ordered = sorted(parent_ids, key=sort_key)
    return _prefer_named_year(ordered, records, query)


def _prefer_named_year(
    ordered: list[str], records: dict[str, dict[str, Any]], query: str
) -> list[str]:
    """Swap a named-year copy ahead of its sibling, without moving other documents."""
    from rag.versions import query_names_document

    if not query or len(ordered) < 2:
        return ordered
    groups: dict[str, list[str]] = {}
    for parent_id in ordered:
        group = (records.get(parent_id) or {}).get("version_group")
        if group:
            groups.setdefault(group, []).append(parent_id)
    out = list(ordered)
    for members in groups.values():
        named = [
            parent_id
            for parent_id in members
            if query_names_document(
                query,
                (records.get(parent_id) or {}).get("source_path") or "",
                (records.get(parent_id) or {}).get("review_date"),
            )
        ]
        others = [parent_id for parent_id in members if parent_id not in named]
        if not named or not others:
            continue
        named_at = out.index(named[0])
        other_at = out.index(others[0])
        if other_at < named_at:
            out[named_at], out[other_at] = out[other_at], out[named_at]
    return out


def _parent_order(chunks: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        parent_id = cast(str, chunk["parent_id"])
        if parent_id in seen:
            continue
        seen.add(parent_id)
        ordered.append(parent_id)
    return ordered


def _rank_scores(parent_ids: list[str]) -> dict[str, float]:
    return {parent_id: 1.0 / (index + 1) for index, parent_id in enumerate(parent_ids)}


def _from_parent_ids(
    index: Index,
    parent_ids: list[str],
    chunks: dict[str, dict[str, Any]],
    child_scores: dict[str, float],
    scores: dict[str, float],
    confident: bool,
    query: str = "",
) -> Retrieval:
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks.values():
        by_parent.setdefault(cast(str, chunk["parent_id"]), []).append(chunk)
    child_for: dict[str, str] = {}
    for parent_id in parent_ids[:MAX_PARENTS]:
        child_for[parent_id] = _best_child(parent_id, by_parent.get(parent_id, []), child_scores)
    return _emit(index, parent_ids[:MAX_PARENTS], scores, child_for, confident, query, focus=True)


def _best_child(
    parent_id: str, parent_chunks: list[dict[str, Any]], child_scores: dict[str, float]
) -> str:
    del parent_id
    best_id = ""
    best_score: float | None = None
    for chunk in parent_chunks:
        score = child_scores.get(chunk["chunk_id"], 0.0)
        if best_score is None or score > best_score:
            best_id = cast(str, chunk["chunk_id"])
            best_score = score
    return best_id


def _emit(
    index: Index,
    parent_ids: list[str],
    scores: dict[str, float],
    child_for: dict[str, str],
    confident: bool,
    query: str = "",
    focus: bool = True,
) -> Retrieval:
    from rag.versions import downrank_superseded

    records = index.get_parents(parent_ids)
    hits: list[Hit] = []
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
    downrank_superseded(hits, query, focus=focus)
    return Retrieval(hits=hits)
