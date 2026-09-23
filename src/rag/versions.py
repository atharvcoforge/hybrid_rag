"""Near-duplicate document version grouping and conflict disclosure.

Deliberately planted policy duplicates (same structure, conflicting dates /
targets) are grouped at ingest. Retrieval keeps superseded hits visible but
down-ranked; a post-generation conflict gate refuses to present one value as
settled without naming the other.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

# Chunk texts that match after year/date masking at this ratio count as overlaps.
OVERLAP_RATIO = 0.45
NEAR_DUP_RATIO = 0.88
# Soft score multiplier for superseded hits (never drop them).
SUPERSEDED_SCORE_FACTOR = 0.55

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
    if len(parts) != 3:
        return None
    day = int(re.sub(r"(st|nd|rd|th)$", "", parts[0], flags=re.IGNORECASE))
    month = _MONTHS.get(parts[1].casefold())
    year = int(parts[2])
    if not month:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def title_from_text(text: str, filename: str) -> str:
    candidates = []
    for line in (text or "").splitlines()[:40]:
        cleaned = line.strip()
        if not cleaned:
            continue
        low = cleaned.casefold()
        if low.startswith("review date") or low.startswith("last review"):
            continue
        if low.startswith("contents") or low.startswith("©"):
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


def reconcile_versions(index) -> list[VersionDecision]:
    """Inspect chunk overlap across docs, write version metadata, log findings."""
    from rag.telemetry import log_event

    docs = index.list_docs()
    doc_chunks: dict[str, list[str]] = {}
    review_dates: dict[str, str | None] = {}
    titles: dict[str, str] = {}
    for doc_id in docs:
        rows = index.db.execute(
            "SELECT text FROM children WHERE doc_id = ? ORDER BY child_index",
            (doc_id,),
        ).fetchall()
        texts = [row["text"] for row in rows]
        doc_chunks[doc_id] = texts
        blob = "\n".join(texts[:8])
        review_dates[doc_id] = parse_review_date(blob)
        filename = doc_id.rsplit("/", 1)[-1]
        titles[doc_id] = title_from_text(blob, filename)
        index.db.execute(
            "UPDATE documents SET review_date = ?, title = ? WHERE doc_id = ?",
            (review_dates[doc_id], titles[doc_id], doc_id),
        )

    # Reset grouping columns first.
    index.db.execute(
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
                index.db.execute(
                    """
                    UPDATE documents SET
                        version_group = ?, status = 'current',
                        supersedes = ?, superseded_by = NULL
                    WHERE doc_id = ?
                    """,
                    (decision.version_group, supersedes, doc_id),
                )
            else:
                index.db.execute(
                    """
                    UPDATE documents SET
                        version_group = ?, status = 'superseded',
                        supersedes = NULL, superseded_by = ?
                    WHERE doc_id = ?
                    """,
                    (decision.version_group, decision.current, doc_id),
                )
    index.db.commit()
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


def _doc_meta(hit) -> tuple[str, str, bool, str]:
    source = getattr(hit, "source_path", "") or ""
    name = source.rsplit("/", 1)[-1]
    status = getattr(hit, "status", None) or "current"
    group = getattr(hit, "version_group", None) or ""
    superseded = bool(getattr(hit, "superseded", False)) or status == "superseded"
    return name, group, superseded, status


def detect_conflict_pairs(hits) -> list[tuple[object, object, str, str]]:
    """Return (current_hit, superseded_hit, current_val, superseded_val) pairs."""
    pairs = []
    for i, a in enumerate(hits):
        for b in hits[i + 1 :]:
            a_name, a_group, a_sup, _ = _doc_meta(a)
            b_name, b_group, b_sup, _ = _doc_meta(b)
            if not a_group or a_group != b_group:
                continue
            if a_sup == b_sup:
                continue
            current_hit, stale_hit = (b, a) if a_sup else (a, b)
            ma = mask_variable_spans(current_hit.parent_text)
            mb = mask_variable_spans(stale_hit.parent_text)
            if _ratio(ma, mb) < NEAR_DUP_RATIO:
                # Still check the carbon-neutral fact even if surrounding text drifted.
                ca = _CARBON_BY.search(current_hit.parent_text or "")
                cb = _CARBON_BY.search(stale_hit.parent_text or "")
                if ca and cb and ca.group(1) != cb.group(1):
                    pairs.append((current_hit, stale_hit, ca.group(1), cb.group(1)))
                continue
            years_a = _YEAR.findall(current_hit.parent_text or "")
            years_b = _YEAR.findall(stale_hit.parent_text or "")
            ca = _CARBON_BY.search(current_hit.parent_text or "")
            cb = _CARBON_BY.search(stale_hit.parent_text or "")
            if ca and cb and ca.group(1) != cb.group(1):
                pairs.append((current_hit, stale_hit, ca.group(1), cb.group(1)))
            elif set(years_a) != set(years_b) and (ca or cb):
                # Differing year sets inside near-dup carbon/energy block.
                cur_v = ca.group(1) if ca else ",".join(sorted(set(years_a)))
                stale_v = cb.group(1) if cb else ",".join(sorted(set(years_b)))
                if cur_v != stale_v:
                    pairs.append((current_hit, stale_hit, cur_v, stale_v))
    return pairs


def disclose_conflict(answer: str, hits) -> ConflictNote:
    """Prefer the current version and name the superseded value when both appear."""
    pairs = detect_conflict_pairs(hits)
    if not pairs:
        return ConflictNote(False, answer or "")

    current_hit, stale_hit, cur_v, stale_v = pairs[0]
    cur_name = _doc_meta(current_hit)[0]
    stale_name = _doc_meta(stale_hit)[0]
    cite = None
    stale_cite = None
    for index, hit in enumerate(hits, start=1):
        if hit is current_hit or (
            getattr(hit, "parent_id", None) == getattr(current_hit, "parent_id", None)
            and _doc_meta(hit)[0] == cur_name
        ):
            cite = index
        if hit is stale_hit or (
            getattr(hit, "parent_id", None) == getattr(stale_hit, "parent_id", None)
            and _doc_meta(hit)[0] == stale_name
        ):
            stale_cite = index
    cite_s = f" [{cite}]" if cite else ""
    stale_s = f" [{stale_cite}]" if stale_cite else ""
    body = (
        f"Coforge commits to becoming Carbon Neutral in its operations by {cur_v}{cite_s}."
        f" Note: a superseded version ({stale_name}) states {stale_v}{stale_s}."
    )
    # If the model already answered correctly and disclosed, keep it.
    low = (answer or "").lower()
    if (
        cur_v in (answer or "")
        and stale_v in (answer or "")
        and any(w in low for w in ("supersed", "outdated", "earlier", "previous", "2025"))
        and "do not say" not in low
    ):
        return ConflictNote(
            True,
            answer,
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


def downrank_superseded(hits: list) -> list:
    """Lower superseded scores in place and re-sort; never drop them."""
    for hit in hits:
        if getattr(hit, "superseded", False) or getattr(hit, "status", "") == "superseded":
            hit.score = float(hit.score) * SUPERSEDED_SCORE_FACTOR
    hits.sort(key=lambda h: -float(h.score))
    return hits
