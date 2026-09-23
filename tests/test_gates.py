from rag.gates import attach_citations, check_all, check_form, check_grounding
from rag.models import Hit


def _hit(text, derived=False):
    hit = Hit(
        parent_id="p",
        parent_text=text,
        heading_path="",
        source_path="Carbon_Reduction_Plan.pdf",
        file_sha256="abc",
        page_start=1,
        page_end=1,
        start_char=0,
        end_char=len(text),
        child_id="c",
        score=0.9,
        confident=True,
    )
    hit.derived = derived
    return hit


def test_decline_without_a_citation_is_withheld():
    result = check_form("The documents do not say.", 2)
    assert result.ok
    assert result.state == "withheld"
    assert result.reason == "declined"


def test_form_requires_in_range_citation():
    assert check_form("India was 14,644 [1].", 1).ok
    assert not check_form("India was 14,644.", 1).ok
    assert not check_form("India was 14,644 [9].", 1).ok
    assert not check_form("", 1).ok
    assert not check_form("ok [1]", 1, finish_reason="length").ok


def test_grounding_catches_fabricated_figures():
    hits = [_hit("India's FY24 baseline was 14,644 tCO2e.")]
    ok = check_grounding("India's FY24 baseline was 14,644 tCO2e [1].", hits)
    assert ok.ok
    bad = check_grounding("India's FY24 baseline was 14,844 tCO2e [1].", hits)
    assert not bad.ok
    assert bad.reason == "unsupported_figure"
    assert "14,844" in bad.unsupported


def test_derived_passage_cannot_sole_support_a_figure():
    hits = [_hit("India's FY24 baseline was 14,644 tCO2e.", derived=True)]
    bad = check_grounding("India's FY24 baseline was 14,644 tCO2e [1].", hits)
    assert not bad.ok


def test_supported_uncited_fact_gains_a_citation():
    hits = [_hit("Publication date: 10 October 2025")]
    assert attach_citations("10 October 2025", hits) == "10 October 2025 [1]"
    result = check_all("10 October 2025", hits)
    assert result.ok
    assert result.answer == "10 October 2025 [1]"
    assert result.state == "verified"


def test_ordinal_date_in_the_passage_supports_the_plain_date():
    hits = [_hit("Publication date: 10th October 2025")]
    result = check_all("10 October 2025", hits)
    assert result.ok
    assert result.answer is not None and "[1]" in result.answer


def test_two_word_claim_is_cited_when_the_year_was_in_the_question():
    hits = [_hit('committed to being "Water Positive by 2040"')]
    result = check_all(
        'The policy sets the ambition to become "Water Positive" by 2040.',
        hits,
        query="What water ambition does the environmental policy set for 2040?",
    )
    assert result.ok
    assert result.answer is not None and "[1]" in result.answer


def test_percent_with_a_space_still_supports_the_figure():
    hits = [_hit("GHG reduction target of 5 % year on year")]
    result = check_all("The policy sets a GHG reduction target of 5%.", hits)
    assert result.ok
    assert result.answer is not None and "[1]" in result.answer


def test_paraphrase_is_cited_when_the_name_and_title_are_in_the_passage():
    hits = [_hit("Approved by John Speight, President and Executive Director.")]
    answer = "John Speight holds the titles of President and Executive Director."
    result = check_all(answer, hits, query="Who signed the environmental policy?")
    assert result.ok
    assert result.answer is not None and "[1]" in result.answer


def test_bare_integer_in_the_passage_is_cited():
    hits = [_hit("| Scope 1 | | | 413 |")]
    result = check_all("The India's baseline Scope 1 figure in tCO2e is 413.", hits, query="India baseline Scope 1 tCO2e")
    assert result.ok
    assert result.answer is not None
    assert "[1]" in result.answer


def test_unit_repeated_from_the_question_does_not_block_a_cited_number():
    hits = [_hit("| Total emissions | | | 14,644 |")]
    query = "What was the baseline total emissions in tCO2e for FY24 in India?"
    answer = "The baseline total emissions in tCO2e for FY24 in India were 14,644."
    result = check_all(answer, hits, query=query)
    assert result.ok
    assert result.answer is not None
    assert "14,644" in result.answer
    assert "[1]" in result.answer


def test_invented_figure_stays_uncited_and_is_withheld():
    hits = [_hit("Publication date: 10 October 2025")]
    result = check_all("The plan was published on 10 October 2099.", hits)
    assert not result.ok
    assert result.reason == "missing_citation"


def test_decline_is_not_rewritten_with_a_citation():
    hits = [_hit("Publication date: 10 October 2025")]
    result = check_all("The documents do not say.", hits)
    assert result.ok
    assert result.reason == "declined"
    assert result.answer is None


def test_check_all_passes_a_clean_answer():
    hits = [_hit("Publication date: 10 October 2025")]
    result = check_all("The plan was published on 10 October 2025 [1].", hits)
    assert result.ok
    assert result.state == "verified"


def test_check_all_does_not_rewrite_signer_when_policies_conflict():
    signer = _hit("Signed on behalf of the Supplier: John Speight")
    current = _hit('Carbon Neutral in our operations by 2040')
    current.version_group = "env"
    current.status = "current"
    stale = _hit('Carbon Neutral in our operations by 2050')
    stale.version_group = "env"
    stale.superseded = True
    stale.status = "superseded"
    stale.source_path = "Environmental_Sustainability_Policy_2025.pdf"
    current.source_path = "Environmental_Sustainability_Policy_2026.pdf"
    result = check_all(
        "Signed by John Speight [1].",
        [signer, current, stale],
        query="Who signed the Carbon Reduction Plan?",
    )
    assert result.state != "conflict"
    assert result.conflict is None
