"""Post-generation citation gates.

Cheapest first. Each gate can stop the answer. Derived (OCR/VLM) blocks
cannot be the sole support for a factual claim. Conflict disclosure runs
before form/literal so a planted version clash is never silent.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rag.models import Hit

_CITE = re.compile(r"\[(\d+)\]")
_DECLINE = re.compile(
    r"do not say|not in the documents|documents do not|not stated|not mentioned|no information",
    re.IGNORECASE,
)
# Numbers, dates, percentages, currency, unit-bearing quantities.
_QUANTITY = re.compile(
    r"""
    (?<![A-Za-z0-9])
    (?:
        \d{1,3}(?:,\d{3})+(?:\.\d+)?   # 14,644 or 1,265.5
      | \d+\.\d+                        # 914.38
      | \d+%                            # 50%
      | \d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|
            July|August|September|October|November|December)\s+\d{4}
      | (?:FY|fy)\s?\d{2}(?:\d{2})?
      | tCO2e|KWp|PPN\s*\d+/\d+
      | \d{2,}                         # 413, 40 — not Scope 1
    )
    (?![A-Za-z0-9])
    """,
    re.VERBOSE | re.IGNORECASE,
)


@dataclass
class GateResult:
    ok: bool
    reason: str = ""
    coverage: float = 1.0
    groundedness: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    unsupported: list[str] = field(default_factory=list)
    state: str = "verified"  # verified | partly_verified | withheld | conflict
    conflict: dict[str, str | int | None] | None = None
    answer: str | None = None


def _answer_quantities(text: str) -> list[str]:
    quantities = [_norm_quantity(match.group(0)) for match in _QUANTITY.finditer(text)]
    for year in re.findall(r"(?<!\d)(20\d{2})(?!\d)", text):
        if not any(year in token for token in quantities):
            quantities.append(year)
    return quantities


def _asked_quantities(query: str) -> set[str]:
    """Units and figures the question already states are not new claims."""
    return set(_answer_quantities(query or ""))


def attach_citations(answer: str, hits: Sequence[Hit], query: str = "") -> str:
    """Cite passages that support the claim. Leave declines and unsupported text alone."""
    original = (answer or "").strip()
    if not original or not hits:
        return original
    text = original
    cites = [int(match.group(1)) for match in _CITE.finditer(text)]
    if cites and all(1 <= cite <= len(hits) for cite in cites):
        return original
    if cites:
        text = _CITE.sub("", text).strip()
    if not text or (_DECLINE.search(text) and not _CITE.search(text)):
        return original if not cites else text

    eligible = []
    for index, hit in enumerate(hits, start=1):
        if getattr(hit, "derived", False):
            continue
        blob = _norm_quantity(getattr(hit, "parent_text", "") or "")
        stale = bool(getattr(hit, "superseded", False)) or getattr(hit, "status", "") == "superseded"
        eligible.append((index, blob, stale))
    if not eligible:
        return text

    quantities = [token for token in _answer_quantities(text) if token not in _asked_quantities(query)]
    if quantities:
        covered: set[str] = set()
        chosen: list[int] = []
        ranked = sorted(eligible, key=lambda item: (item[2], -sum(q in item[1] for q in quantities), item[0]))
        for index, blob, _stale in ranked:
            owned = [token for token in quantities if token in blob and token not in covered]
            if not owned:
                continue
            chosen.append(index)
            covered.update(token for token in quantities if token in blob)
            if set(quantities) <= covered:
                break
        if set(quantities) <= covered and chosen:
            return _append_cites(text, chosen)
        return text

    span = _norm_quantity(text)
    ordered = sorted(eligible, key=lambda item: (item[2], item[0]))
    words = _content_tokens(span)
    asked_words = set(_content_tokens(_norm_quantity(query)))
    phrases = []
    for left, right in itertools.pairwise(words):
        phrase = f"{left} {right}"
        if len(phrase) < 8:
            continue
        novel = left not in asked_words or right not in asked_words
        phrases.append((0 if novel else 1, phrase))
    for _novel, phrase in phrases:
        for index, blob, _stale in ordered:
            if _phrase_in(blob, phrase):
                return _append_cites(text, [index])
    if len(span) >= 8:
        for index, blob, _stale in ordered:
            if _phrase_in(blob, span):
                return _append_cites(text, [index])
    asked = set(_content_tokens(_norm_quantity(query)))
    tokens = _content_tokens(span)
    distinctive = [token for token in tokens if token not in asked] or tokens
    if len(distinctive) >= 2:
        best_index = None
        best_n = 0
        best_stale = True
        for index, blob, stale in ordered:
            found = sum(1 for token in distinctive if _phrase_in(blob, token))
            if found > best_n or (found == best_n and best_stale and not stale):
                best_n = found
                best_index = index
                best_stale = stale
        if best_index is not None and best_n >= 2 and best_n / len(distinctive) >= 0.5:
            return _append_cites(text, [best_index])
    return text


_CONTENT_STOP = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "on", "by", "for", "and", "or", "is",
        "was", "were", "be", "been", "being", "does", "do", "did", "what", "when",
        "which", "who", "how", "from", "with", "that", "this", "these", "those",
        "has", "have", "had", "its", "it", "as", "at", "about", "into", "over",
        "after", "before", "not", "no", "yes", "are", "our", "their", "than", "more",
    }
)


def _content_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9%.-]*", text.casefold())
    return [word for word in words if word not in _CONTENT_STOP and len(word) >= 3]


def _phrase_in(blob: str, phrase: str) -> bool:
    if not phrase:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", blob) is not None


def _append_cites(text: str, indexes: list[int]) -> str:
    marks = "".join(f" [{index}]" for index in sorted(set(indexes)))
    return text.rstrip() + marks


def check_form(answer: str, n_hits: int, finish_reason: str | None = None) -> GateResult:
    text = (answer or "").strip()
    if not text:
        return GateResult(False, "empty_answer", state="withheld")
    if finish_reason == "length":
        return GateResult(False, "truncated", state="withheld")
    cites = [int(match.group(1)) for match in _CITE.finditer(text)]
    if not cites and _DECLINE.search(text):
        return GateResult(True, "declined", state="withheld")
    if not cites:
        return GateResult(False, "missing_citation", state="withheld")
    if any(cite < 1 or cite > n_hits for cite in cites):
        return GateResult(False, "citation_out_of_range", state="withheld")
    return GateResult(True, "")


def _norm_quantity(value: str) -> str:
    text = value.strip().casefold().replace("\u2212", "-")
    text = text.replace("per cent", "%").replace("percent", "%")
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    text = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _passage_blob(hits: Sequence[Hit], cited: set[int]) -> str:
    parts = []
    for index, hit in enumerate(hits, start=1):
        if index not in cited:
            continue
        if getattr(hit, "derived", False):
            continue
        parts.append(hit.parent_text)
    return _norm_quantity("\n".join(parts))


def check_grounding(answer: str, hits: Sequence[Hit], query: str = "") -> GateResult:
    cites = {int(match.group(1)) for match in _CITE.finditer(answer or "")}
    blob = _passage_blob(hits, cites)
    asked = _asked_quantities(query)
    unsupported = []
    for token in _answer_quantities(answer or ""):
        if token in asked:
            continue
        if token and token not in blob:
            unsupported.append(token)
    if unsupported:
        return GateResult(
            False,
            "unsupported_figure",
            coverage=0.0,
            unsupported=unsupported,
            state="withheld",
        )
    return GateResult(True, "", coverage=1.0, state="verified")


def coverage_state(coverage: float) -> str:
    if coverage >= 0.9:
        return "verified"
    if coverage >= 0.5:
        return "partly_verified"
    return "withheld"


def check_conflict(answer: str, hits: Sequence[Hit], query: str = "") -> GateResult:
    from rag.versions import disclose_conflict

    note = disclose_conflict(answer, hits, query)
    if not note.fired:
        return GateResult(True, "", state="verified")
    return GateResult(
        True,
        "version_conflict",
        state="conflict",
        conflict={
            "current_doc": note.current_doc,
            "superseded_doc": note.superseded_doc,
            "current_value": note.current_value,
            "superseded_value": note.superseded_value,
            "cite": note.cite,
            "version_group": note.version_group,
        },
        answer=note.answer,
    )


def check_all(
    answer: str,
    hits: Sequence[Hit],
    finish_reason: str | None = None,
    entailment: Callable[[str, Sequence[Hit]], GateResult] | None = None,
    query: str = "",
) -> GateResult:
    from rag.telemetry import span

    attached = attach_citations(answer, hits, query)
    with span("gate_form"):
        form = check_form(attached, len(hits), finish_reason)
    grounded = GateResult(True, "")
    if form.reason == "declined":
        with span("gate_literal", skipped=True, reason="declined"):
            pass
    elif form.ok:
        with span("gate_literal"):
            grounded = check_grounding(attached, hits, query)
    else:
        with span("gate_literal", skipped=True, reason="form_failed"):
            pass

    result: GateResult
    if form.reason == "declined" or not form.ok:
        result = form
    elif not grounded.ok:
        result = grounded
    elif entailment is None:
        with span("gate_entail", skipped=True, reason="no_entailment_model") as entail_span:
            entail_span["degraded"] = True
        with span("gate_coverage", coverage=1.0):
            result = GateResult(
                True,
                "",
                coverage=1.0,
                groundedness=1.0,
                citation_precision=1.0,
                citation_recall=1.0,
                state="verified",
            )
    else:
        with span("gate_entail"):
            result = entailment(attached, hits)
        with span(
            "gate_coverage",
            coverage=getattr(result, "coverage", None),
        ):
            pass

    with span("gate_conflict") as conflict_span:
        conflict = check_conflict(attached, hits, query)
        if conflict.conflict:
            conflict_span["reason"] = "version_conflict"
            return GateResult(
                True,
                "version_conflict",
                coverage=getattr(result, "coverage", 1.0) or 1.0,
                groundedness=1.0,
                citation_precision=1.0,
                citation_recall=1.0,
                state="conflict",
                conflict=conflict.conflict,
                answer=conflict.answer,
            )
        conflict_span["skipped"] = True
        conflict_span["reason"] = "no_conflict"
    if result.ok and result.reason != "declined":
        result.answer = attached
    return result
