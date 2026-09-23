"""Post-generation citation gates.

Cheapest first. Each gate can stop the answer. Derived (OCR/VLM) blocks
cannot be the sole support for a factual claim. Conflict disclosure runs
before form/literal so a planted version clash is never silent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITE = re.compile(r"\[(\d+)\]")
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
    conflict: dict | None = None
    answer: str | None = None


def check_form(answer: str, n_hits: int, finish_reason: str | None = None) -> GateResult:
    text = (answer or "").strip()
    if not text:
        return GateResult(False, "empty_answer", state="withheld")
    if finish_reason == "length":
        return GateResult(False, "truncated", state="withheld")
    cites = [int(match.group(1)) for match in _CITE.finditer(text)]
    if not cites:
        return GateResult(False, "missing_citation", state="withheld")
    if any(cite < 1 or cite > n_hits for cite in cites):
        return GateResult(False, "citation_out_of_range", state="withheld")
    return GateResult(True, "")


def _norm_quantity(value: str) -> str:
    text = value.strip().casefold().replace("\u2212", "-")
    text = text.replace("per cent", "%").replace("percent", "%")
    text = re.sub(r"\s+", " ", text)
    return text


def _passage_blob(hits, cited: set[int]) -> str:
    parts = []
    for index, hit in enumerate(hits, start=1):
        if index not in cited:
            continue
        if getattr(hit, "derived", False):
            continue
        parts.append(hit.parent_text)
    return _norm_quantity("\n".join(parts))


def check_grounding(answer: str, hits) -> GateResult:
    cites = {int(match.group(1)) for match in _CITE.finditer(answer or "")}
    blob = _passage_blob(hits, cites)
    unsupported = []
    for match in _QUANTITY.finditer(answer or ""):
        token = _norm_quantity(match.group(0))
        if token and token not in blob:
            unsupported.append(match.group(0))
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


def check_conflict(answer: str, hits) -> GateResult:
    from rag.versions import disclose_conflict

    note = disclose_conflict(answer, hits)
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


def check_all(answer: str, hits, finish_reason: str | None = None, entailment=None) -> GateResult:
    from rag.telemetry import span

    with span("gate_form"):
        form = check_form(answer, len(hits), finish_reason)
    grounded = GateResult(True, "")
    if form.ok:
        with span("gate_literal"):
            grounded = check_grounding(answer, hits)
    else:
        with span("gate_literal", skipped=True, reason="form_failed"):
            pass

    result: GateResult
    if not form.ok:
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
            result = entailment(answer, hits)
        with span(
            "gate_coverage",
            coverage=getattr(result, "coverage", None),
        ):
            pass

    with span("gate_conflict") as conflict_span:
        conflict = check_conflict(answer if result.ok else (result.answer or answer), hits)
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
    return result
