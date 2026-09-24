from rag.models import Hit
from rag.versions import (
    chunk_overlap_ratio,
    disclose_conflict,
    downrank_superseded,
    group_documents,
    mask_variable_spans,
    parse_review_date,
    title_match_boost,
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
    note = disclose_conflict(
        "The documents do not say.",
        [stale, current],
        "By when does Coforge commit to becoming carbon neutral in its operations?",
    )
    assert note.fired
    assert "2040" in note.answer
    assert "2050" in note.answer
    assert "Environmental_Sustainability_Policy_2025.pdf" in note.answer
    assert "superseded" in note.answer.lower()
    assert "do not say" not in note.answer.lower()


_BOTH = (
    "Review Date – 10th March {year_review} Last Review – 1st April {year_last}. "
    'committed to become "Carbon Neutral in our operations by {carbon}". '
    "Procure 10% of electricity from green sources by {green10} and increase the share to ~50% by {green50}. "
    "electric vehicles (EVs) in fleet for employee commute to 10% by {ev10} and to 50% by {ev50}. "
    "© {copy}"
)


def _policy_pair():
    stale = _hit(
        _BOTH.format(
            year_review="2025",
            year_last="2024",
            carbon="2050",
            green10="2027",
            green50="2040",
            ev10="2030",
            ev50="2045",
            copy="2025",
        ),
        "Environmental_Sustainability_Policy_2025.pdf",
        superseded=True,
    )
    current = _hit(
        _BOTH.format(
            year_review="2026",
            year_last="2025",
            carbon="2040",
            green10="2025",
            green50="2030",
            ev10="2025",
            ev50="2040",
            copy="2026",
        ),
        "Environmental_Sustainability_Policy_2026.pdf",
        superseded=False,
    )
    current.parent_id = "current1"
    stale.parent_id = "stale001"
    return stale, current


def test_an_answered_question_keeps_its_text_and_names_the_superseded_file():
    stale, current = _policy_pair()
    note = disclose_conflict(
        "Signed by John Speight [1].",
        [stale, current],
        "Who signed the Carbon Reduction Plan?",
    )
    assert note.fired
    assert "John Speight" in note.answer
    assert "Environmental_Sustainability_Policy_2025.pdf" in note.answer


def test_a_year_that_also_appears_later_is_still_disclosed():
    current = _hit(
        "Carbon neutral by 2040. Green share by 2030. Fleet by 2040.",
        "Environmental_Sustainability_Policy_2026.pdf",
    )
    stale = _hit(
        "Carbon neutral by 2050. Green share by 2040. Fleet by 2045.",
        "Environmental_Sustainability_Policy_2025.pdf",
        superseded=True,
    )
    note = disclose_conflict("The target is 2040.", [current, stale], "carbon neutral")
    assert "2050" in note.answer
    assert "2040" in note.answer


def test_disclosure_follows_the_number_the_answer_already_used():
    electricity_old = _hit(
        "Procure 10% of electricity from green sources by 2027.",
        "Environmental_Sustainability_Policy_2025.pdf",
        superseded=True,
    )
    electricity_new = _hit(
        "Procure 10% of electricity from green sources by 2025.",
        "Environmental_Sustainability_Policy_2026.pdf",
    )
    carbon_old = _hit(
        "Carbon Neutral in our operations by 2050.",
        "Environmental_Sustainability_Policy_2025.pdf",
        superseded=True,
    )
    carbon_new = _hit(
        "Carbon Neutral in our operations by 2040.",
        "Environmental_Sustainability_Policy_2026.pdf",
    )
    note = disclose_conflict(
        "Coforge commits to becoming carbon neutral by 2050.",
        [electricity_old, electricity_new, carbon_old, carbon_new],
        "By when does Coforge commit to becoming carbon neutral?",
    )
    assert "2040" in note.answer
    assert "2050" in note.answer
    assert "2027" not in note.answer


def test_review_date_conflict_names_march_dates_not_carbon_year():
    stale, current = _policy_pair()
    note = disclose_conflict(
        "The documents do not say.",
        [stale, current],
        "What is the Review Date on the Environmental Sustainability Policy?",
    )
    assert note.fired
    assert "10th March 2026" in note.answer
    assert "10th March 2025" in note.answer
    assert "Environmental_Sustainability_Policy_2025.pdf" in note.answer


def test_a_question_that_repeats_a_title_outranks_a_higher_score():
    env = _hit(
        "Energy Optimization and green energy by 2025",
        "Environmental_Sustainability_Policy_2026.pdf",
        score=14.0,
    )
    carbon = _hit(
        "10% green energy by 2025 in the carbon plan",
        "Carbon_Reduction_Plan.pdf",
        score=9.0,
    )
    env.title = "Environmental Sustainability Policy"
    carbon.title = "Carbon Reduction Plan"
    assert title_match_boost("unrelated question about leave", "Carbon Reduction Plan") == 1.0
    assert title_match_boost(
        "What does the environmental policy require?",
        "Sustainability Policy",
        "Environmental_Sustainability_Policy_2026.pdf",
    ) == title_match_boost("environmental policy", "Environmental Policy")
    ordered = downrank_superseded(
        [env, carbon],
        "What share of green energy does the carbon plan target by 2025?",
    )
    assert ordered[0].source_path == "Carbon_Reduction_Plan.pdf"
    assert ordered[0].score == 9.0


def test_contents_page_ranks_below_the_section_that_answers():
    contents = _hit(
        "Contents Scope ................ 1 Energy ................ 2",
        "Environmental_Sustainability_Policy_2026.pdf",
        score=6.5,
    )
    section = _hit(
        "Review Date – 10th March 2026",
        "Environmental_Sustainability_Policy_2026.pdf",
        score=6.0,
    )
    ordered = downrank_superseded(
        [contents, section],
        "What is the Review Date on the Environmental Sustainability Policy?",
    )
    assert ordered[0].parent_text.startswith("Review Date")


def test_downrank_keeps_superseded_but_sorts_below_current():
    stale = _hit("stale", "Environmental_Sustainability_Policy_2025.pdf", superseded=True, score=1.0)
    current = _hit("cur", "Environmental_Sustainability_Policy_2026.pdf", superseded=False, score=0.9)
    ordered = downrank_superseded([stale, current])
    assert ordered[0].source_path == "Environmental_Sustainability_Policy_2026.pdf"
    assert any(h.source_path.endswith("2025.pdf") for h in ordered)
