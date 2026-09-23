from rag.gates import (
    _answer_quantities,
    _phrase_in,
    attach_citations,
    check_all,
    check_grounding,
    coverage_state,
)
from rag.models import Hit


def _hit(text, **extra):
    hit = Hit("p", text, "H", "a.md", "abc", 1, 1, 0, len(text), "c", 1.0, False)
    for key, value in extra.items():
        setattr(hit, key, value)
    return hit


def test_a_year_glued_to_a_letter_is_still_collected():
    assert "2040" in _answer_quantities("see x2040 in the note")


def test_attach_citations_repairs_strips_and_skips_derived_hits():
    hit = _hit("the latch is open")
    assert attach_citations("see [9]", [hit]) == "see"
    assert attach_citations("not in the documents [9]", [hit]) == "not in the documents"
    derived = _hit("the latch is open", derived=True)
    assert attach_citations("the latch is open", [derived]) == "the latch is open"


def test_quantities_pick_every_supporting_passage_and_stop_when_covered():
    first = _hit("revenue was 10%")
    second = _hit("headcount was 40")
    cited = attach_citations("10% and 40", [first, second])
    assert "[1]" in cited and "[2]" in cited
    one = attach_citations("10% and 40", [_hit("revenue was 10% and headcount was 40")])
    assert one.endswith("[1]")


def test_empty_answer_and_a_decline_after_a_bad_cite_are_left_alone():
    assert attach_citations("", [_hit("text")]) == ""
    assert attach_citations("not stated here", []) == "not stated here"
    hit = _hit("the latch is open")
    assert "not stated" in attach_citations("not stated [9]", [hit])


def test_phrase_and_token_overlap_choose_a_current_passage():
    assert attach_citations("abc xyz", [_hit("abc xyz stays")]).endswith("[1]")
    span_hit = _hit("prefix abc def ghi suffix")
    assert attach_citations("abc def ghi", [span_hit]).endswith("[1]")
    scattered = _hit("ghi is here and abc is there and def is last")
    assert attach_citations("abc def ghi", [scattered]).endswith("[1]")
    weak = attach_citations("alpha beta", [_hit("alpha only here")])
    assert weak == "alpha beta"
    current = _hit("alpha xx gamma")
    stale = _hit("alpha yy gamma", superseded=True)
    chosen = attach_citations("alpha gamma", [current, stale])
    assert chosen.endswith("[1]")
    assert _phrase_in("blob", "") is False


def test_coverage_thresholds_and_grounding_with_an_entailment_model():
    assert coverage_state(0.9) == "verified"
    assert coverage_state(0.5) == "partly_verified"
    assert coverage_state(0.49) == "withheld"
    hits = [_hit("revenue was 10%")]
    grounded = check_grounding("revenue was 99% [1]", hits)
    assert grounded.ok is False
    result = check_all("revenue was 99% [1]", hits)
    assert result.ok is False

    def entail(_answer, _hits):
        from rag.gates import GateResult

        return GateResult(True, "", coverage=0.7, state="partly_verified")

    checked = check_all("the latch is open [1]", [_hit("the latch is open")], entailment=entail)
    assert checked.ok is True


def test_a_fired_conflict_replaces_the_answer(monkeypatch):
    from rag.versions import ConflictNote

    monkeypatch.setattr(
        "rag.versions.disclose_conflict",
        lambda *_args, **_kwargs: ConflictNote(
            True,
            "rewritten",
            current_doc="now.md",
            superseded_doc="old.md",
            current_value="2050",
            superseded_value="2040",
            cite=1,
            version_group="carbon",
        ),
    )
    result = check_all("carbon in 2050 [1]", [_hit("carbon neutrality by 2050")])
    assert result.conflict
    assert result.answer == "rewritten"
