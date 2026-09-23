"""Table ladder and structure gate (F-12, §05)."""

from rag.layout import (
    extract_page_tables,
    looks_borderless,
    score_table,
)


def test_score_table_accepts_consistent_numeric_columns():
    rows = [
        ["Region", "tCO2e", "Share"],
        ["India", "14,644", "39%"],
        ["UK", "8,688", "23%"],
    ]
    ok, reason = score_table(rows)
    assert ok, reason
    assert reason == ""


def test_score_table_rejects_ragged_columns():
    rows = [
        ["A", "B", "C"],
        ["1", "2"],
        ["3", "4", "5"],
    ]
    ok, reason = score_table(rows)
    assert not ok
    assert reason == "ragged_columns"


def test_score_table_rejects_numeric_header():
    rows = [
        ["1", "2", "3"],
        ["a", "b", "c"],
    ]
    ok, reason = score_table(rows)
    assert not ok
    assert reason == "numeric_header"


def test_score_table_rejects_non_numeric_under_numeric_header():
    rows = [
        ["Site", "tCO2e"],
        ["India", "fourteen"],
    ]
    ok, reason = score_table(rows)
    assert not ok
    assert reason == "non_numeric_cell"


def test_looks_borderless_needs_aligned_whitespace_columns():
    prose = "Just a sentence.\nAnother line of prose.\nNothing columnar here."
    assert not looks_borderless(prose)

    columns = (
        "Region          Scope 1       Scope 2\n"
        "India           14,644        1,265\n"
        "United Kingdom  8,688         914\n"
        "USA             5,543         852\n"
    )
    assert looks_borderless(columns)


class _FakeTable:
    def __init__(self, rows, bbox=(10, 10, 200, 100)):
        self._rows = rows
        self.bbox = bbox

    def extract(self):
        return self._rows


class _FakePage:
    def __init__(self, ruled=None, text_strategy=None, text=""):
        self._ruled = ruled or []
        self._text_strategy = text_strategy or []
        self._text = text
        self.width = 612
        self.height = 792

    def find_tables(self, table_settings=None):
        if table_settings and table_settings.get("vertical_strategy") == "text":
            return list(self._text_strategy)
        return list(self._ruled)

    def extract_text(self):
        return self._text


def test_ladder_uses_ruled_tables_first():
    page = _FakePage(
        ruled=[_FakeTable([["A", "B"], ["1", "2"]])],
        text_strategy=[_FakeTable([["X", "Y"], ["9", "8"]])],
    )
    tables = extract_page_tables(page)
    assert len(tables) == 1
    assert tables[0].rung == 1
    assert "A | B" in tables[0].text


def test_ladder_falls_to_text_strategy_when_ruled_empty():
    page = _FakePage(
        ruled=[],
        text_strategy=[
            _FakeTable([["Region", "Scope 1", "Scope 2"], ["India", "14,644", "1,265"]])
        ],
        text=(
            "Region          Scope 1       Scope 2\n"
            "India           14,644        1,265\n"
            "United Kingdom  8,688         914\n"
            "USA             5,543         852\n"
        ),
    )
    tables = extract_page_tables(page)
    assert len(tables) == 1
    assert tables[0].rung == 2
    assert "14,644" in tables[0].text


def test_ladder_escalates_to_layout_when_text_strategy_fails_gate():
    ragged = _FakeTable([["A", "B", "C"], ["1", "2"], ["3", "4", "5", "6"]])
    page = _FakePage(
        ruled=[],
        text_strategy=[ragged],
        text=(
            "Region          Scope 1       Scope 2\n"
            "India           14,644        1,265\n"
            "United Kingdom  8,688         914\n"
            "USA             5,543         852\n"
        ),
    )
    called = []

    def layout_extract(_page):
        called.append(True)
        return [[["Region", "tCO2e"], ["India", "14,644"]]]

    tables = extract_page_tables(page, layout_extract=layout_extract)
    assert called == [True]
    assert len(tables) == 1
    assert tables[0].rung == 3
    assert tables[0].ok
    assert "14,644" in tables[0].text


def test_failed_gate_still_emits_flagged_prose_table():
    ragged = _FakeTable([["A", "B"], ["only-one"]])
    page = _FakePage(ruled=[ragged])
    tables = extract_page_tables(page, layout_extract=lambda _p: [])
    assert len(tables) == 1
    assert not tables[0].ok
    assert tables[0].flagged
    assert "only-one" in tables[0].text
