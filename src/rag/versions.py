"""Near-duplicate document version grouping and conflict disclosure.

Deliberately planted policy duplicates (same structure, conflicting dates /
targets) are grouped at ingest. Retrieval keeps superseded hits visible but
down-ranked; a post-generation conflict gate refuses to present one value as
settled without naming the other.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from rag.models import Hit
    from rag.store import Index

# Chunk texts that match after year/date masking at this ratio count as overlaps.
OVERLAP_RATIO = 0.45
NEAR_DUP_RATIO = 0.88
# Soft score multiplier for superseded hits (never drop them).
SUPERSEDED_SCORE_FACTOR = 0.55
# A question that repeats a stored document title should not lose to a sibling file.
TITLE_MATCH_BOOST = 1.65
# Contents pages list every heading, so they win overlap and do not hold the fact.
TOC_FACTOR = 0.6

_REVIEW = re.compile(
    r"Review\s+Date\s*[–—\-:]\s*(\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(20\d{2})\b")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def mask_variable_spans(text: str) -> str:
    """Collapse years and ordinal dates so near-duplicate policy text aligns."""
    out = _YEAR.sub("YEAR", text)
    out = re.sub(
        r"\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December)\s+YEAR",
        "DATE",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\s+", " ", out).strip().casefold()
    return out


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Cheap token Jaccard — good enough for long near-identical policy chunks.
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def chunk_overlap_ratio(texts_a: list[str], texts_b: list[str]) -> float:
    if not texts_a or not texts_b:
        return 0.0
    masks_b = [mask_variable_spans(t) for t in texts_b]
    hits = 0
    for text in texts_a:
        ma = mask_variable_spans(text)
        if any(_ratio(ma, mb) >= NEAR_DUP_RATIO for mb in masks_b):
            hits += 1
    return hits / len(texts_a)


def parse_review_date(text: str) -> str | None:
    match = _REVIEW.search(text or "")
    if not match:
        return None
    raw = match.group(1)
    parts = raw.split()
    day = int(re.sub(r"(st|nd|rd|th)$", "", parts[0], flags=re.IGNORECASE))
    month = _MONTHS[parts[1].casefold()]
    year = int(parts[2])
    return f"{year:04d}-{month:02d}-{day:02d}"


def title_from_text(text: str, filename: str) -> str:
    candidates = []
    for line in (text or "").splitlines()[:40]:
        cleaned = line.strip()
        if not cleaned:
            continue
        low = cleaned.casefold()
        if low.startswith(("review date", "last review")):
            continue
        if low.startswith(("contents", "©")):
            break
        if len(cleaned) < 8 or len(cleaned) > 120:
            continue
        candidates.append(cleaned)
        if "policy" in low or "plan" in low:
            return cleaned
    if candidates:
        return max(candidates, key=len)
    stem = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip()
    return stem or filename


@dataclass
class VersionDecision:
    version_group: str
    members: list[str]
    current: str
    superseded: list[str]
    review_dates: dict[str, str | None] = field(default_factory=dict)


def group_documents(
    doc_chunks: dict[str, list[str]],
    review_dates: dict[str, str | None],
    *,
    threshold: float = OVERLAP_RATIO,
) -> list[VersionDecision]:
    """Connected components of near-duplicate docs → version groups."""
    docs = sorted(doc_chunks)
    parent: dict[str, str] = {d: d for d in docs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(docs):
        for b in docs[i + 1 :]:
            ab = chunk_overlap_ratio(doc_chunks[a], doc_chunks[b])
            ba = chunk_overlap_ratio(doc_chunks[b], doc_chunks[a])
            if max(ab, ba) >= threshold:
                union(a, b)

    clusters: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        clusters[find(doc)].append(doc)

    decisions = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        dated = [(review_dates.get(d) or "", d) for d in members]
        dated.sort(reverse=True)  # ISO dates sort; missing date sorts last
        current = dated[0][1]
        superseded = [d for _, d in dated[1:]]
        # Stable group id from sorted member stems.
        slug = "vg_" + "_".join(
            re.sub(r"[^a-z0-9]+", "", m.lower().rsplit("/", 1)[-1].rsplit(".", 1)[0])[:24]
            for m in members
        )[:80]
        decisions.append(
            VersionDecision(
                version_group=slug,
                members=members,
                current=current,
                superseded=superseded,
                review_dates={m: review_dates.get(m) for m in members},
            )
        )
    return decisions


def reconcile_versions(index: Index) -> list[VersionDecision]:
    """Inspect chunk overlap across docs, write version metadata, log findings."""
    from rag.telemetry import log_event

    db = cast(sqlite3.Connection, index.db)
    docs = index.list_docs()
    doc_chunks: dict[str, list[str]] = {}
    review_dates: dict[str, str | None] = {}
    titles: dict[str, str] = {}
    for doc_id in docs:
        rows = db.execute(
            "SELECT text FROM children WHERE doc_id = ? ORDER BY child_index",
            (doc_id,),
        ).fetchall()
        texts = [row["text"] for row in rows]
        doc_chunks[doc_id] = texts
        blob = "\n".join(texts[:8])
        review_dates[doc_id] = parse_review_date(blob)
        filename = doc_id.rsplit("/", 1)[-1]
        titles[doc_id] = title_from_text(blob, filename)
        db.execute(
            "UPDATE documents SET review_date = ?, title = ? WHERE doc_id = ?",
            (review_dates[doc_id], titles[doc_id], doc_id),
        )

    # Reset grouping columns first.
    db.execute(
        """
        UPDATE documents SET
            version_group = NULL,
            status = 'current',
            supersedes = NULL,
            superseded_by = NULL
        """
    )

    decisions = group_documents(doc_chunks, review_dates)
    for decision in decisions:
        log_event(
            "version_group_found",
            version_group=decision.version_group,
            current=decision.current,
            superseded=decision.superseded,
            review_dates=decision.review_dates,
            level="warning",
        )
        for doc_id in decision.members:
            if doc_id == decision.current:
                supersedes = ",".join(decision.superseded) if decision.superseded else None
                db.execute(
                    """
                    UPDATE documents SET
                        version_group = ?, status = 'current',
                        supersedes = ?, superseded_by = NULL
                    WHERE doc_id = ?
                    """,
                    (decision.version_group, supersedes, doc_id),
                )
            else:
                db.execute(
                    """
                    UPDATE documents SET
                        version_group = ?, status = 'superseded',
                        supersedes = NULL, superseded_by = ?
                    WHERE doc_id = ?
                    """,
                    (decision.version_group, decision.current, doc_id),
                )
    db.commit()
    return decisions


@dataclass
class ConflictNote:
    fired: bool
    answer: str
    current_doc: str = ""
    superseded_doc: str = ""
    current_value: str = ""
    superseded_value: str = ""
    cite: int | None = None
    version_group: str = ""


_TITLE_STOP = {
    "what", "when", "which", "who", "how", "does", "with", "from", "that", "this",
    "have", "been", "will", "should", "under", "about", "their", "there", "into",
    "the", "and", "for", "are", "was", "were", "you", "your",
}
_ABSTAIN = ("do not say", "not in the documents", "documents do not", "not stated")


def _terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9-]{2,}", (text or "").casefold())
    return [word for word in words if word not in _TITLE_STOP]


def title_match_boost(query: str, title: str, source_path: str = "") -> float:
    """Boost a hit when the question repeats words from the title or the filename."""
    stem = (source_path or "").rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("_", " ")
    title_terms = set(_terms(f"{title} {stem}"))
    if len(title_terms) < 2:
        return 1.0
    overlap = {term for term in _terms(query) if term in title_terms}
    if len(overlap) >= 2:
        return TITLE_MATCH_BOOST
    return 1.0


def query_names_version(query: str, source_path: str = "", review_date: str | None = None) -> bool:
    """True when the question names this document's year."""
    text = (query or "").casefold()
    if not text:
        return False
    name = (source_path or "").rsplit("/", 1)[-1]
    years = set(re.findall(r"20\d{2}", name))
    if review_date and review_date[:4].isdigit():
        years.add(review_date[:4])
    return any(year in text for year in years)


def query_names_document(query: str, source_path: str = "", review_date: str | None = None) -> bool:
    """True when the question names this file, not merely a target year such as 'by 2025'."""
    if not query_names_version(query, source_path, review_date):
        return False
    return bool(re.search(r"\b(policy|version|document|edition)\b", query or "", re.IGNORECASE))


def looks_like_toc(text: str) -> bool:
    return (text or "").lstrip()[:8].casefold() == "contents"


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text or ""))


def _is_stale(hit: Hit) -> bool:
    return bool(getattr(hit, "superseded", False) or getattr(hit, "status", "") == "superseded")


def _aligned_texts(heading_a: str, text_a: str, heading_b: str, text_b: str) -> bool:
    left = (heading_a or "").strip()
    right = (heading_b or "").strip()
    if left and right and left == right:
        return True
    return (
        _ratio(mask_variable_spans(text_a), mask_variable_spans(text_b)) >= NEAR_DUP_RATIO
    )


def _window(text: str, token: str) -> str:
    at = (text or "").find(token)
    if at < 0:
        return token
    start = max(0, at - 48)
    if start > 0 and not text[start - 1].isspace() and not text[start].isspace():
        space = text.find(" ", start, at)
        if space != -1:
            start = space + 1
    end = min(len(text), at + len(token) + 8)
    return " ".join(text[start:end].split())


def _windows(text: str, tokens: list[str], focus: str = "") -> str:
    parts: list[str] = []
    seen: set[str] = set()
    focus_nums = _numbers(focus)
    ordered = sorted(tokens, key=lambda token: (token not in focus_nums, text.find(token)))
    for token in ordered[:8]:
        span = _window(text, token)
        if span not in seen:
            seen.add(span)
            parts.append(span)
    return "; ".join(parts)


def _unique(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _number_diff(text_a: str, text_b: str) -> tuple[list[str], list[str]]:
    """Numbers that differ between two near-duplicate passages.

    Set difference misses a year that moved: 2040 can be the new carbon target
    and also appear later as the old date of a different target. Compare the
    number sequences in order when the passages still line up.
    """
    if _ratio(mask_variable_spans(text_a), mask_variable_spans(text_b)) < NEAR_DUP_RATIO:
        return [], []
    seq_a = _NUMBER.findall(text_a or "")
    seq_b = _NUMBER.findall(text_b or "")
    if seq_a and len(seq_a) == len(seq_b):
        changed_a = [left for left, right in zip(seq_a, seq_b) if left != right]
        changed_b = [right for left, right in zip(seq_a, seq_b) if left != right]
        if changed_a:
            return _unique(changed_a), _unique(changed_b)
    nums_a, nums_b = set(seq_a), set(seq_b)
    only_a = sorted(nums_a - nums_b, key=lambda token: text_a.find(token))
    only_b = sorted(nums_b - nums_a, key=lambda token: text_b.find(token))
    return only_a, only_b


def values_differ(text_a: str, text_b: str) -> bool:
    only_a, only_b = _number_diff(text_a, text_b)
    return bool(only_a and only_b)


def _hit_name(hit: Hit) -> str:
    source = getattr(hit, "source_path", "") or ""
    return source.rsplit("/", 1)[-1]


def _conflicting_pair(
    hits: Sequence[Hit],
    focus: str = "",
) -> tuple[Hit, Hit, list[str], list[str]] | None:
    groups: dict[str, list[Hit]] = {}
    for hit in hits:
        group = getattr(hit, "version_group", None) or ""
        if group:
            groups.setdefault(group, []).append(hit)
    found: list[tuple[Hit, Hit, list[str], list[str]]] = []
    for members in groups.values():
        currents = [hit for hit in members if not _is_stale(hit)]
        stales = [hit for hit in members if _is_stale(hit)]
        for current in currents:
            for stale in stales:
                if not _aligned_texts(
                    current.heading_path,
                    current.parent_text,
                    stale.heading_path,
                    stale.parent_text,
                ):
                    continue
                only_current, only_stale = _number_diff(current.parent_text, stale.parent_text)
                if only_current and only_stale:
                    found.append((current, stale, only_current, only_stale))
    if not found:
        return None
    focus_nums = _numbers(focus)
    if focus_nums:
        for item in found:
            numbers = set(item[2]) | set(item[3])
            if focus_nums & numbers:
                return item
    return found[0]


def disclose_conflict(answer: str, hits: Sequence[Hit], query: str = "") -> ConflictNote:
    """When a current passage and its superseded twin disagree, say so."""
    text = answer or ""
    pair = _conflicting_pair(hits, f"{query}\n{text}")
    if pair is None:
        return ConflictNote(False, text)
    current, stale, only_current, only_stale = pair
    stale_name = _hit_name(stale)
    current_name = _hit_name(current)
    cite = list(hits).index(current) + 1
    low = text.lower()
    named = stale_name.lower() in low
    has_current = any(value in text for value in only_current)
    has_stale = any(value in text for value in only_stale)
    note = ConflictNote(
        True,
        text,
        current_doc=current_name,
        superseded_doc=stale_name,
        current_value=only_current[0],
        superseded_value=only_stale[0],
        cite=cite,
        version_group=getattr(current, "version_group", "") or "",
    )
    if named and has_current and has_stale:
        return note
    sentence = (
        f"The current document states {_windows(current.parent_text, only_current, text)}. "
        f"Note: a superseded version ({stale_name}) states {_windows(stale.parent_text, only_stale, text)}."
    )
    abstained = (not text.strip()) or any(marker in low for marker in _ABSTAIN)
    note.answer = sentence if abstained else text.rstrip() + "\n" + sentence
    return note


def aligned_peer_records(index: Index, hits: list[Hit]) -> list[dict[str, Any]]:
    """The other version of a retrieved section, when its dates or numbers differ."""
    if not hits or not hasattr(index, "list_doc_records") or not hasattr(index, "parent_records"):
        return []
    docs = list(index.list_doc_records())
    present = {hit.parent_id for hit in hits}
    wanted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        group = getattr(hit, "version_group", None)
        if not group:
            continue
        hit_stale = _is_stale(hit)
        for record in docs:
            if record.get("version_group") != group:
                continue
            peer_stale = (record.get("status") or "current") == "superseded"
            if peer_stale == hit_stale:
                continue
            already = [
                other
                for other in hits
                if (getattr(other, "version_group", None) or "") == group and _is_stale(other) == peer_stale
            ]
            if any(
                _aligned_texts(hit.heading_path, hit.parent_text, other.heading_path, other.parent_text)
                and values_differ(hit.parent_text, other.parent_text)
                for other in already
            ):
                continue
            doc_id = record.get("id")
            if not doc_id:
                continue
            best: dict[str, Any] | None = None
            best_ratio = -1.0
            for parent in index.parent_records(doc_id):
                parent_id = parent.get("parent_id")
                if not parent_id or parent_id in present or parent_id in seen:
                    continue
                heading = parent.get("heading_path") or ""
                text = parent.get("text") or ""
                same_heading = bool(
                    (hit.heading_path or "").strip() and (hit.heading_path or "").strip() == heading.strip()
                )
                ratio = _ratio(mask_variable_spans(hit.parent_text), mask_variable_spans(text))
                if not same_heading and ratio < NEAR_DUP_RATIO:
                    continue
                if not values_differ(hit.parent_text, text):
                    continue
                if same_heading or ratio > best_ratio:
                    best = parent
                    best_ratio = 1.0 if same_heading else ratio
                    if same_heading:
                        break
            if best is None:
                continue
            parent_id = str(best["parent_id"])
            seen.add(parent_id)
            present.add(parent_id)
            wanted.append(best)
    return wanted


def downrank_superseded(hits: list[Hit], query: str = "", focus: bool = True) -> list[Hit]:
    """Lower superseded scores in place and re-sort; never drop them."""
    del focus
    for hit in hits:
        named = query_names_document(
            query,
            getattr(hit, "source_path", "") or "",
            getattr(hit, "review_date", None),
        )
        if _is_stale(hit) and not named:
            hit.score = float(hit.score) * SUPERSEDED_SCORE_FACTOR
        if looks_like_toc(getattr(hit, "parent_text", "") or ""):
            hit.score = float(hit.score) * TOC_FACTOR
        # The title boost reorders. It must not scale the score: sigmoid
        # scores live on 0–1, and a 1.65× factor pushes them above every tau.
    hits.sort(
        key=lambda hit: -(
            float(hit.score)
            * title_match_boost(
                query,
                getattr(hit, "title", None) or "",
                getattr(hit, "source_path", "") or "",
            )
        )
    )
    _prefer_named_copy(hits, query)
    return hits


def _prefer_named_copy(hits: list[Hit], query: str) -> None:
    """Move a copy whose year the question names ahead of its sibling."""
    if not query or len(hits) < 2:
        return
    groups: dict[str, list[Hit]] = {}
    for hit in hits:
        group = getattr(hit, "version_group", None)
        if group:
            groups.setdefault(group, []).append(hit)
    for members in groups.values():
        named = [
            hit
            for hit in members
            if query_names_document(
                query,
                getattr(hit, "source_path", "") or "",
                getattr(hit, "review_date", None),
            )
        ]
        others = [hit for hit in members if hit not in named]
        if not named or not others:
            continue
        named_at = hits.index(named[0])
        other_at = hits.index(others[0])
        if other_at < named_at:
            hits[named_at], hits[other_at] = hits[other_at], hits[named_at]


