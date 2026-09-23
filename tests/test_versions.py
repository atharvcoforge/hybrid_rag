from rag.models import Hit
from rag.versions import (
    chunk_overlap_ratio,
    disclose_conflict,
    downrank_superseded,
    group_documents,
    mask_variable_spans,
    parse_review_date,
)


def test_parse_review_date():
    assert parse_review_date("Review Date – 10th March 2025\n") == "2025-03-10"
    assert parse_review_date("Review Date – 10th March 2026\n") == "2026-03-10"


def test_chunk_overlap_masks_year_conflicts():
    a = "We are committed to become Carbon Neutral in our operations by 2050. Procure by 2027."
    b = "We are committed to become Carbon Neutral in our operations by 2040. Procure by 2025."
    assert mask_variable_spans(a) == mask_variable_spans(b)
    assert chunk_overlap_ratio([a], [b]) == 1.0


def test_group_documents_picks_newest_review_date():
    stale = ["Carbon Neutral in our operations by 2050. green sources by 2027."]
    current = ["Carbon Neutral in our operations by 2040. green sources by 2025."]
    other = ["Water positive by 2040 unrelated unique text about irrigation quotas."]
    decisions = group_documents(
        {
            "Environmental_Sustainability_Policy_2025.pdf": stale * 5,
            "Environmental_Sustainability_Policy_2026.pdf": current * 5,
            "Water_Management_Policy.pdf": other * 5,
        },
        {
            "Environmental_Sustainability_Policy_2025.pdf": "2025-03-10",
            "Environmental_Sustainability_Policy_2026.pdf": "2026-03-10",
            "Water_Management_Policy.pdf": "2024-01-01",
        },
    )
    assert len(decisions) == 1
    d = decisions[0]
    assert d.current == "Environmental_Sustainability_Policy_2026.pdf"
    assert d.superseded == ["Environmental_Sustainability_Policy_2025.pdf"]


def _hit(text, source, *, superseded=False, group="env", score=1.0):
    hit = Hit(
        parent_id=source[:8],
        parent_text=text,
        heading_path="",
        source_path=source,
        file_sha256="x",
        page_start=1,
        page_end=1,
        start_char=0,
        end_char=len(text),
        child_id="c",
        score=score,
        confident=False,
    )
    hit.superseded = superseded
    hit.status = "superseded" if superseded else "current"
    hit.version_group = group
    hit.superseded_by = "Environmental_Sustainability_Policy_2026.pdf" if superseded else None
    return hit


def test_disclose_conflict_prefers_2040_and_names_2050():
    stale = _hit(
        'We, at Coforge, are committed to become "Carbon Neutral in our operations by 2050".',
        "Environmental_Sustainability_Policy_2025.pdf",
        superseded=True,
        score=0.9,
    )
    current = _hit(
        'We, at Coforge, are committed to become "Carbon Neutral in our operations by 2040".',
        "Environmental_Sustainability_Policy_2026.pdf",
        superseded=False,
        score=0.9,
    )
    note = disclose_conflict("The documents do not say.", [stale, current])
    assert note.fired
    assert "2040" in note.answer
    assert "2050" in note.answer
    assert "Environmental_Sustainability_Policy_2025.pdf" in note.answer
    assert "superseded" in note.answer.lower()
    assert "do not say" not in note.answer.lower()


def test_downrank_keeps_superseded_but_sorts_below_current():
    stale = _hit("stale", "Environmental_Sustainability_Policy_2025.pdf", superseded=True, score=1.0)
    current = _hit("cur", "Environmental_Sustainability_Policy_2026.pdf", superseded=False, score=0.9)
    ordered = downrank_superseded([stale, current])
    assert ordered[0].source_path == "Environmental_Sustainability_Policy_2026.pdf"
    assert any(h.source_path.endswith("2025.pdf") for h in ordered)
