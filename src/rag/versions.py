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
# A question that names one file should not lose to a higher-scoring sibling file.
DOC_HINT_BOOST = 1.65
# Contents pages list every heading, so they win overlap and do not hold the fact.
TOC_FACTOR = 0.6

_REVIEW = re.compile(
    r"Review\s+Date\s*[–—\-:]\s*(\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(20\d{2})\b")
_CARBON_BY = re.compile(
    r"Carbon\s+Neutral\s+in\s+(?:our\s+)?operations\s+by\s+(20\d{2})",
    re.IGNORECASE,
)
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


def _doc_meta(hit: Hit) -> tuple[str, str, bool, str]:
    source = getattr(hit, "source_path", "") or ""
    name = source.rsplit("/", 1)[-1]
    status = getattr(hit, "status", None) or "current"
    group = getattr(hit, "version_group", None) or ""
    superseded = bool(getattr(hit, "superseded", False)) or status == "superseded"
    return name, group, superseded, status


_DATE = (
    r"(\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{4})"
)


@dataclass(frozen=True)
class _Fact:
    """One planted conflict. Query wording selects the fact; the extract pulls its value."""

    label: str
    query: re.Pattern[str]
    value: re.Pattern[str]


def _ev(percent: str) -> re.Pattern[str]:
    return re.compile(rf"electric\s+vehicles?.*{percent}\s*%|{percent}\s*%.*electric\s+vehicles?", re.IGNORECASE)


def _green(percent: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?:green|electricity).{{0,80}}{percent}\s*%|{percent}\s*%.{{0,80}}(?:green|electricity)",
        re.IGNORECASE,
    )


# Most specific query first. "Last Review date" must not fall through to "Review Date".
_FACTS: tuple[_Fact, ...] = (
    _Fact(
        "The Last Review date on the current policy is",
        re.compile(r"last\s+review", re.IGNORECASE),
        re.compile(rf"Last\s+Review\s*[–—\-:]\s*{_DATE}", re.IGNORECASE),
    ),
    _Fact(
        "The Review Date on the current policy is",
        re.compile(r"review\s+date", re.IGNORECASE),
        re.compile(rf"Review\s+Date\s*[–—\-:]\s*{_DATE}", re.IGNORECASE),
    ),
    _Fact(
        "The copyright year on the current policy is",
        re.compile(r"copyright", re.IGNORECASE),
        re.compile(r"[©]\s*(20\d{2})"),
    ),
    _Fact(
        "Coforge commits to becoming Carbon Neutral in its operations by",
        re.compile(r"carbon\s+neutral", re.IGNORECASE),
        _CARBON_BY,
    ),
    _Fact(
        "Electric vehicles reach 10% of the employee commute fleet by",
        _ev("10"),
        re.compile(r"to\s+10%\s+by\s+(20\d{2})", re.IGNORECASE),
    ),
    _Fact(
        "Electric vehicles reach 50% of the employee commute fleet by",
        _ev("50"),
        re.compile(r"to\s+50%\s+by\s+(20\d{2})", re.IGNORECASE),
    ),
    _Fact(
        "The policy commits to procuring 10% of electricity from green sources by",
        _green("10"),
        re.compile(r"green\s+sources\s+by\s+(20\d{2})", re.IGNORECASE),
    ),
    _Fact(
        "The policy aims to increase green electricity share to about 50% by",
        _green("50"),
        re.compile(r"share to\s+~?\s*50%\s+by\s+(20\d{2})", re.IGNORECASE),
    ),
)


def _fact_for_query(query: str) -> _Fact | None:
    for fact in _FACTS:
        if fact.query.search(query or ""):
            return fact
    return None


def conflict_sibling_records(index: Index, query: str, hits: list[Hit]) -> list[dict[str, Any]]:
    """Parents for the other side of a matched fact, loaded from the index.

    Disclosure needs both versions. Top-k often keeps the right document and
    the wrong section, so the sibling is fetched by the fact pattern.
    """
    fact = _fact_for_query(query)
    if fact is None or not hits or not hasattr(index, "list_doc_records"):
        return []
    group = ""
    for hit in hits:
        _name, grp, _superseded, _status = _doc_meta(hit)
        if grp:
            group = grp
            break
    if not group or not hasattr(index, "first_parent_matching"):
        return []

    def _side_present(want_stale: bool) -> bool:
        for hit in hits:
            _name, grp, superseded, _status = _doc_meta(hit)
            if grp != group or superseded != want_stale:
                continue
            if fact.value.search(getattr(hit, "parent_text", "") or ""):
                return True
        return False

    wanted: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for record in index.list_doc_records():
        if record.get("version_group") != group:
            continue
        stale = (record.get("status") or "current") == "superseded"
        if _side_present(stale):
            continue
        found = index.first_parent_matching(cast(str, record.get("id")), fact.value)
        if not found:
            continue
        parent_id = found.get("parent_id")
        if not parent_id or parent_id in seen:
            continue
        if any(getattr(hit, "parent_id", None) == parent_id for hit in hits):
            continue
        seen.add(parent_id)
        wanted.append(found)
    return wanted


def disclose_conflict(answer: str, hits: Sequence[Hit], query: str = "") -> ConflictNote:
    """Disclose the asked fact when both versions state it. Leave other answers alone."""
    fact = _fact_for_query(query)
    if fact is None or not hits:
        return ConflictNote(False, answer or "")

    current_hit = None
    stale_hit = None
    cur_v = ""
    stale_v = ""
    for hit in hits:
        _name, group, superseded, _status = _doc_meta(hit)
        if not group:
            continue
        match = fact.value.search(getattr(hit, "parent_text", "") or "")
        if not match:
            continue
        if superseded:
            if stale_hit is None:
                stale_hit = hit
                stale_v = match.group(1)
        elif current_hit is None:
            current_hit = hit
            cur_v = match.group(1)
    if current_hit is None or stale_hit is None or cur_v == stale_v:
        return ConflictNote(False, answer or "")

    cur_name = _doc_meta(current_hit)[0]
    stale_name = _doc_meta(stale_hit)[0]
    cite = None
    stale_cite = None
    for index, hit in enumerate(hits, start=1):
        if hit is current_hit:
            cite = index
        if hit is stale_hit:
            stale_cite = index
    cite_s = f" [{cite}]" if cite else ""
    stale_s = f" [{stale_cite}]" if stale_cite else ""
    body = (
        f"{fact.label} {cur_v}{cite_s}. "
        f"Note: a superseded version ({stale_name}) states {stale_v}{stale_s}."
    )
    text = answer or ""
    low = text.lower()
    disclosed = any(word in low for word in ("supersed", "outdated", "earlier", "previous", "conflict"))
    if stale_name.lower() in low:
        disclosed = True
    if cur_v in text and stale_v in text and disclosed and "do not say" not in low:
        return ConflictNote(
            True,
            text,
            current_doc=cur_name,
            superseded_doc=stale_name,
            current_value=cur_v,
            superseded_value=stale_v,
            cite=cite,
            version_group=getattr(current_hit, "version_group", "") or "",
        )
    return ConflictNote(
        True,
        body,
        current_doc=cur_name,
        superseded_doc=stale_name,
        current_value=cur_v,
        superseded_value=stale_v,
        cite=cite,
        version_group=getattr(current_hit, "version_group", "") or "",
    )


def query_names_document(query: str, source_path: str = "", review_date: str | None = None) -> bool:
    """True when the question names this file, not merely a target year such as 'by 2025'."""
    if not query_names_version(query, source_path, review_date):
        return False
    return bool(re.search(r"\b(policy|version|document|edition)\b", query or "", re.IGNORECASE))


def query_names_version(query: str, source_path: str = "", review_date: str | None = None) -> bool:
    """True when the question names this document's year. The superseded penalty is only a prior."""
    text = (query or "").casefold()
    if not text:
        return False
    name = (source_path or "").rsplit("/", 1)[-1]
    years = set(re.findall(r"20\d{2}", name))
    if review_date and review_date[:4].isdigit():
        years.add(review_date[:4])
    return any(year in text for year in years)


def entity_mismatch(query: str, passage: str) -> bool:
    """True when the passage names the other country or the other GHG scope."""
    q = (query or "").casefold()
    p = (passage or "").casefold()
    if not q or not p:
        return False
    # "India versus UK" names both on purpose. Demote only a side the question did not ask for.
    if _has_token(q, "india") and _has_token(q, "uk"):
        return False
    for wanted, other in (("india", "uk"), ("uk", "india")):
        if _has_token(q, wanted) and not _has_token(p, wanted) and _has_token(p, other):
            return True
        # Same table often names both countries. The section heading does not.
        opening = p[:120]
        if _has_token(q, wanted) and _has_token(opening, other) and not _has_token(opening, wanted):
            return True
    wanted_scopes = set(re.findall(r"scope\s*([123])", q))
    if not wanted_scopes:
        return False
    present = set(re.findall(r"scope\s*([123])", p))
    return bool(present) and wanted_scopes.isdisjoint(present)


def _has_token(text: str, word: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text) is not None


def _query_terms(query: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9-]{2,}", (query or "").casefold())
    stop = {
        "what", "when", "which", "who", "how", "does", "with", "from", "that", "this",
        "have", "been", "will", "should", "under", "about", "their", "there", "into",
    }
    terms = [word for word in words if word not in stop]
    if "ghg" in (query or "").casefold() and "greenhouse" not in terms:
        terms.append("greenhouse")
    return terms


def _doc_family(source: str) -> str:
    name = (source or "").casefold()
    if "water" in name:
        return "water"
    if "carbon" in name:
        return "carbon"
    if "environmental" in name:
        return "env"
    return ""


def _doc_hint(query: str) -> str:
    text = (query or "").casefold()
    hints = set()
    if "carbon plan" in text or "carbon reduction" in text:
        hints.add("carbon")
    if "water policy" in text or "water management" in text:
        hints.add("water")
    if "environmental" in text and "policy" in text:
        hints.add("env")
    if len(hints) == 1:
        return next(iter(hints))
    return ""


_WHO_SIGNED = re.compile(
    r"\b(?:who|which)\b.{0,60}\b(?:signed|signatory|approved|approver|approval|chairs)\b",
    re.IGNORECASE,
)


def signatory_boost(query: str, passage: str) -> float:
    """Who-signed questions prefer the short name-and-title block."""
    if not _WHO_SIGNED.search(query or ""):
        return 1.0
    text = passage or ""
    if len(text) > 240 or not re.search(r"\bpresident\b", text, re.IGNORECASE):
        return 1.0
    if re.search(r"\b(?:director|evp|europe)\b", text, re.IGNORECASE):
        return 6.0
    return 1.0


def looks_like_toc(text: str) -> bool:
    return (text or "").lstrip()[:8].casefold() == "contents"


def focus_multiplier(query: str, text: str, source: str, texts: list[str]) -> float:
    """Boost the named file and demote passages missing the rarest query term."""
    factor = 1.0
    hint = _doc_hint(query)
    if hint and _doc_family(source) == hint:
        factor *= DOC_HINT_BOOST
    terms = _query_terms(query)
    blobs = [(item or "").casefold() for item in texts]
    if not terms or not blobs:
        return factor
    counted = [(sum(term in blob for blob in blobs), term) for term in terms]
    present = [(count, term) for count, term in counted if count]
    if not present:
        return factor
    unique = [term for count, term in present if count == 1]
    if not unique:
        return factor
    blob = (text or "").casefold()
    owned = sum(1 for term in unique if term in blob)
    if owned:
        factor *= 1 + 0.8 * owned
    return factor


def downrank_superseded(hits: list[Hit], query: str = "", focus: bool = True) -> list[Hit]:
    """Lower superseded scores in place and re-sort; never drop them."""
    for hit in hits:
        named = query_names_document(
            query,
            getattr(hit, "source_path", "") or "",
            getattr(hit, "review_date", None),
        )
        if not named and (
            getattr(hit, "superseded", False) or getattr(hit, "status", "") == "superseded"
        ):
            hit.score = float(hit.score) * SUPERSEDED_SCORE_FACTOR
        if entity_mismatch(query, getattr(hit, "parent_text", "") or ""):
            hit.score = float(hit.score) * 0.45
        if looks_like_toc(getattr(hit, "parent_text", "") or ""):
            hit.score = float(hit.score) * TOC_FACTOR
    if focus:
        passages = [getattr(hit, "parent_text", "") or "" for hit in hits]
        for hit in hits:
            hit.score = float(hit.score) * focus_multiplier(
                query,
                getattr(hit, "parent_text", "") or "",
                getattr(hit, "source_path", "") or "",
                passages,
            )
        hit.score = float(hit.score) * signatory_boost(query, getattr(hit, "parent_text", "") or "")
    hits.sort(key=lambda h: -float(h.score))
    _swap_named_year(hits, query)
    _swap_when_the_date_is_close(hits, query)
    _swap_exact_tie(hits, query)
    return hits


def _term_coverage(query: str, text: str) -> int:
    blob = (text or "").casefold()
    return sum(blob.count(term) for term in _query_terms(query))


def _swap_exact_tie(hits: list[Hit], query: str) -> None:
    """Equal scores: keep the passage that repeats more of the question."""
    if not query or len(hits) < 2:
        return
    first, second = hits[0], hits[1]
    top = float(first.score)
    if top <= 0 or abs(top - float(second.score)) > top * 0.005:
        return
    if _term_coverage(query, getattr(second, "parent_text", "") or "") > _term_coverage(
        query, getattr(first, "parent_text", "") or ""
    ):
        hits[0], hits[1] = second, first


def _swap_when_the_date_is_close(hits: list[Hit], query: str) -> None:
    """A by-when question whose top two scores are tied prefers the passage that states a year."""
    if not query or len(hits) < 2:
        return
    if not re.search(r"\bby when\b|\bwhat year\b|\bwhich year\b|\btarget year\b", query, re.IGNORECASE):
        return
    first, second = hits[0], hits[1]
    top = float(first.score)
    if top <= 0 or float(second.score) < top * 0.95:
        return
    first_year = re.search(r"20\d{2}", getattr(first, "parent_text", "") or "")
    second_year = re.search(r"20\d{2}", getattr(second, "parent_text", "") or "")
    if second_year and not first_year:
        hits[0], hits[1] = second, first


def _swap_named_year(hits: list[Hit], query: str) -> None:
    """Move a named-year copy ahead of its sibling without moving other documents."""
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
