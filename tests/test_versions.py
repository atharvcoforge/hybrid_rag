from rag.models import Hit
from rag.versions import (
    chunk_overlap_ratio,
    disclose_conflict,
    downrank_superseded,
    entity_mismatch,
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


def test_signer_question_is_not_rewritten_to_carbon_conflict():
    stale, current = _policy_pair()
    note = disclose_conflict(
        "Signed by John Speight [1].",
        [stale, current],
        "Who signed the Carbon Reduction Plan?",
    )
    assert not note.fired
    assert note.answer == "Signed by John Speight [1]."
    assert "2040" not in note.answer


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
    assert "2040" not in note.answer
    assert "Environmental_Sustainability_Policy_2025.pdf" in note.answer


def test_named_file_outranks_a_higher_scoring_other_document():
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
    ordered = downrank_superseded(
        [env, carbon],
        "What share of green energy does the carbon plan target by 2025?",
    )
    assert ordered[0].source_path == "Carbon_Reduction_Plan.pdf"


def test_rarest_query_term_promotes_the_passage_that_contains_it():
    declaration = _hit(
        "Declaration and Sign Off. This carbon plan reports emissions.",
        "Carbon_Reduction_Plan.pdf",
        score=16.0,
    )
    protocol = _hit(
        "The carbon plan follows the Greenhouse Gas Protocol.",
        "Carbon_Reduction_Plan.pdf",
        score=8.0,
    )
    other = _hit(
        "The carbon plan supplier name is Coforge.",
        "Carbon_Reduction_Plan.pdf",
        score=7.0,
    )
    ordered = downrank_superseded(
        [declaration, protocol, other],
        "Which GHG protocol does the carbon plan cite for reporting?",
    )
    assert "Protocol" in ordered[0].parent_text


def test_a_comparison_that_names_both_countries_is_not_a_mismatch():
    assert not entity_mismatch(
        "How do the India baseline total and the UK baseline total compare?",
        "India operations total 14,644",
    )
    assert entity_mismatch("What is India's baseline Scope 1 figure?", "UK operations only")


def test_exact_tie_prefers_the_passage_that_repeats_the_question():
    vague = _hit("community programs and renewable energy", "Carbon_Reduction_Plan.pdf", score=0.0320)
    direct = _hit(
        "The carbon footprint partner is named Carbon Footprint.",
        "Carbon_Reduction_Plan.pdf",
        score=0.0320,
    )
    ordered = downrank_superseded(
        [vague, direct],
        "Which partner helps with carbon footprint management?",
        focus=False,
    )
    assert "partner" in ordered[0].parent_text


def test_by_when_prefers_a_close_passage_that_states_a_year():
    prose = _hit(
        "Water risk is assessed at owned facilities.",
        "Water_Management_Policy.pdf",
        score=0.0328,
    )
    table = _hit(
        "| Focus Area | Target Year |\n| Owned facilities | 2026 |",
        "Water_Management_Policy.pdf",
        score=0.0323,
    )
    ordered = downrank_superseded(
        [prose, table],
        "By when will water risk assessments cover owned facilities?",
        focus=False,
    )
    assert "2026" in ordered[0].parent_text


def test_who_signed_prefers_the_short_title_block():
    prose = _hit(
        "Declaration and Sign Off. This Carbon Reduction Plan has been completed in accordance with the standard.",
        "Carbon_Reduction_Plan.pdf",
        score=0.04,
    )
    other = _hit(
        "The carbon reduction plan supplier name is Coforge.",
        "Carbon_Reduction_Plan.pdf",
        score=0.03,
    )
    block = _hit(
        "John Speight\nPresident and Head of Europe (EVP)",
        "Carbon_Reduction_Plan.pdf",
        score=0.02,
    )
    ordered = downrank_superseded(
        [prose, other, block],
        "Who signed the Carbon Reduction Plan?",
    )
    assert "Speight" in ordered[0].parent_text


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
