from rag.models import Child, Hit, Parent
from rag.store import SqliteStore
from rag.versions import (
    _doc_family,
    _doc_hint,
    _ratio,
    _swap_exact_tie,
    _swap_when_the_date_is_close,
    chunk_overlap_ratio,
    conflict_sibling_records,
    disclose_conflict,
    group_documents,
    parse_review_date,
    reconcile_versions,
    signatory_boost,
    title_from_text,
)


def _hit(text, source, group=None, superseded=False, parent_id="p"):
    return Hit(
        parent_id,
        text,
        "H",
        source,
        "abc",
        1,
        1,
        0,
        len(text),
        "c",
        1.0,
        False,
        superseded=superseded,
        version_group=group,
        status="superseded" if superseded else "current",
    )


def test_ratio_dates_titles_and_overlap_edges():
    assert _ratio("", "word") == 0.0
    assert _ratio("same", "same") == 1.0
    assert _ratio("   ", "word") == 0.0
    assert parse_review_date("Review Date: 1 January") is None
    assert parse_review_date("Review Date: 1 Notamonth 2024") is None
    assert parse_review_date("Review Date: 1st January 2024") == "2024-01-01"
    titled = "\n\nReview Date: 1 January 2024\nContents\nA real policy title\n"
    assert title_from_text("The carbon policy\n", "file.md") == "The carbon policy"
    assert title_from_text(titled, "file.md")
    assert title_from_text("© 2020\nshort", "my_policy.md")
    assert chunk_overlap_ratio([], ["a"]) == 0.0
    same = ["The carbon policy sets the same operational target for every site."]
    grouped = group_documents(
        {"a.md": same, "b.md": same, "c.md": same},
        {"a.md": "2024-01-01", "b.md": "2025-01-01", "c.md": "2023-01-01"},
        threshold=0.1,
    )
    assert grouped and grouped[0].current == "b.md"


def test_reconcile_writes_a_version_group(tmp_path):
    store = SqliteStore(tmp_path / "idx", "model", "rev", 3)
    store.open()
    try:
        text = "Carbon reduction policy " + ("target " * 40)
        for doc, pid in (("old.md", "p1"), ("new.md", "p2")):
            child = Child(pid + "c", pid, doc, text, text, "H", "prose", 0, 10, 1, 1, 0, 0, 2)
            parent = Parent(pid, doc, text, "H", "prose", 0, 10, 1, 1, 0, 2)
            store.upsert([child], [parent], [[0.1, 0.2, 0.3, 0.4]], {
                "source_path": doc,
                "filename": doc,
                "mime": "text/plain",
                "file_sha256": pid,
                "pipeline_version": 3,
            })
        decisions = reconcile_versions(store)
        assert decisions
        assert decisions[0].superseded
    finally:
        store.close()


def test_conflict_siblings_skip_incomplete_records():
    current = "Carbon Neutral in operations by 2050"
    stale = "Carbon Neutral in operations by 2040"
    hit = _hit(current, "new.md", group="g", parent_id="p-new")
    assert conflict_sibling_records(object(), "hello", [hit]) == []
    assert conflict_sibling_records(object(), "when is carbon neutral", []) == []

    class Index:
        def list_doc_records(self):
            return [
                {"id": "other.md", "version_group": "else", "status": "current"},
                {"id": "now.md", "version_group": "g", "status": "current"},
                {"id": "blank.md", "version_group": "g", "status": "superseded"},
                {"id": "dup.md", "version_group": "g", "status": "superseded"},
                {"id": "gone.md", "version_group": "g", "status": "superseded"},
                {"id": "old.md", "version_group": "g", "status": "superseded"},
            ]

        def first_parent_matching(self, doc_id, _pattern):
            if doc_id == "blank.md":
                return {"text": "Carbon Neutral in operations by 2040"}
            if doc_id == "dup.md":
                return {"parent_id": "p-new"}
            if doc_id == "old.md":
                return {"parent_id": "p-old", "text": stale}
            return None

    ungrouped = _hit("Carbon Neutral in operations by 2050", "loose.md")
    assert conflict_sibling_records(Index(), "when is carbon neutral", [ungrouped]) == []
    both = conflict_sibling_records(
        Index(),
        "when is carbon neutral",
        [
            hit,
            _hit(stale, "old-copy.md", group="g", superseded=True, parent_id="already"),
        ],
    )
    assert both == []
    found = conflict_sibling_records(
        Index(),
        "when is carbon neutral",
        [
            _hit("preamble without a year", "notes.md", group="g"),
            hit,
            _hit("unrelated section", "notes.md", group="other"),
        ],
    )
    assert [row["parent_id"] for row in found] == ["p-old"]

    class NoMatch:
        def list_doc_records(self):
            return [{"id": "x", "version_group": "g", "status": "current"}]

    assert conflict_sibling_records(NoMatch(), "when is carbon neutral", [hit]) == []


def test_disclosure_and_rank_swaps():
    current = _hit("Carbon Neutral in operations by 2050", "new.md", group="g", parent_id="new")
    stale = _hit(
        "Carbon Neutral in operations by 2040",
        "old.md",
        group="g",
        superseded=True,
        parent_id="old",
    )
    again = _hit(
        "Carbon Neutral in operations by 2040",
        "older.md",
        group="g",
        superseded=True,
        parent_id="older",
    )
    note = disclose_conflict(
        "2050",
        [_hit("ungrouped", "loose.md"), _hit("no year here", "mid.md", group="g"), current, stale, again],
        "when is carbon neutral",
    )
    assert note.fired
    same = disclose_conflict("2050", [current, current], "when is carbon neutral")
    assert same.fired is False
    already = disclose_conflict(
        "2050 and 2040 in the superseded old.md",
        [current, stale],
        "when is carbon neutral",
    )
    assert already.fired
    assert disclose_conflict("plain", [], "hello").fired is False

    assert _doc_family("water-policy.md") == "water"
    assert _doc_family("carbon-plan.md") == "carbon"
    assert _doc_family("environmental-policy.md") == "env"
    assert _doc_family("other.md") == ""
    assert _doc_hint("carbon plan and water policy") == ""
    assert signatory_boost("who signed this", "Jane Doe, President and Director, Europe") == 6.0
    assert signatory_boost("who signed this", "Jane Doe, President") == 1.0
    assert signatory_boost("who signed this", "x" * 300) == 1.0

    quiet = [_hit("alpha", "a.md"), _hit("beta", "b.md")]
    quiet[0].score = 1.0
    quiet[1].score = 1.0
    _swap_exact_tie(quiet, "alpha")
    _swap_when_the_date_is_close(quiet, "by when")
    both_years = [_hit("target 2040", "a.md"), _hit("target 2050", "b.md")]
    both_years[0].score = 1.0
    both_years[1].score = 1.0
    _swap_when_the_date_is_close(both_years, "by when is it")
    tied = [_hit("no year here at all", "a.md"), _hit("the target year is 2050", "b.md")]
    tied[0].score = 1.0
    tied[1].score = 1.0
    _swap_when_the_date_is_close(tied, "by when is the target")
    assert tied[0].parent_id == "b.md" or "2050" in tied[0].parent_text
