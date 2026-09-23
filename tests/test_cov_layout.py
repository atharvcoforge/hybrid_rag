"""Branch coverage for rag.layout."""

from rag.layout import _from_find, _is_numeric, extract_page_tables, score_table


class _Table:
    def __init__(self, rows, boom=False):
        self._rows = rows
        self.boom = boom
        self.bbox = (0, 0, 10, 10)

    def extract(self):
        if self.boom:
            raise RuntimeError("bad table")
        return self._rows


class _Page:
    def __init__(self, tables=None, text="", type_error_once=False):
        self._tables = tables or []
        self._text = text
        self._calls = 0
        self.type_error_once = type_error_once
        self.width = 612
        self.height = 792

    def find_tables(self, table_settings=None):
        self._calls += 1
        if self.type_error_once and self._calls == 1:
            raise TypeError("legacy signature")
        if table_settings and table_settings.get("vertical_strategy") == "text":
            raise TypeError("no kwargs")
        return list(self._tables)

    def extract_text(self):
        return self._text


def test_score_table_empty_column_and_mixed_numeric_cells():
    ok, reason = score_table(
        [
            ["Site", "Notes"],
            ["India", ""],
            ["UK", ""],
        ]
    )
    assert ok and reason == ""
    ok, reason = score_table(
        [
            ["Site", "Count"],
            ["India", "1"],
            ["UK", "two"],
        ]
    )
    assert not ok and reason == "non_numeric_cell"
    assert _is_numeric("") is False
    assert _is_numeric("   ") is False
    assert _is_numeric("14,644") is True


def test_ladder_edges_for_missing_layout_and_bad_tables():
    ragged = _Table([["A", "B"], ["only"]])
    flagged = extract_page_tables(_Page(tables=[ragged]))
    assert flagged and flagged[0].flagged

    def explode(_page):
        raise RuntimeError("docling down")

    still = extract_page_tables(_Page(tables=[ragged]), layout_extract=explode)
    assert still and still[0].flagged

    def empty_rows(_page):
        return [[["", ""], ["", ""]], [["Region", "tCO2e"], ["India", "14,644"]]]

    escalated = extract_page_tables(_Page(tables=[ragged]), layout_extract=empty_rows)
    assert any(table.rung == 3 and table.ok for table in escalated)

    columns = (
        "Region          Scope 1       Scope 2\n"
        "India           14,644        1,265\n"
        "United Kingdom  8,688         914\n"
        "USA             5,543         852\n"
    )
    assert extract_page_tables(_Page(text=columns)) == []

    legacy = _Page(type_error_once=True, text="just prose")
    assert _from_find(legacy, rung=1, settings=None) == []

    class _Boom:
        def find_tables(self, table_settings=None):
            raise RuntimeError("no tables")

    assert _from_find(_Boom(), rung=1, settings=None) == []

    bad = _Table(None, boom=True)
    empty = _Table([])
    blank = _Table([["", None], ["  ", ""]])
    good = _Table([["A", "B"], ["1", "2"]])
    found = _from_find(_Page(tables=[bad, empty, blank, good]), rung=1, settings=None)
    assert len(found) == 1
    assert "A | B" in found[0].text


def test_find_tables_errors_are_an_empty_page():
    class Boom:
        def find_tables(self, table_settings=None):
            raise RuntimeError("pdfplumber failed")

    assert _from_find(Boom(), rung=0, settings=None) == []
