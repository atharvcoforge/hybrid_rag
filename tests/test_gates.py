from rag.gates import check_all, check_form, check_grounding
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


def test_check_all_passes_a_clean_answer():
    hits = [_hit("Publication date: 10 October 2025")]
    result = check_all("The plan was published on 10 October 2025 [1].", hits)
    assert result.ok
    assert result.state == "verified"
