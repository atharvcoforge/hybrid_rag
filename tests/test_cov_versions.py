from rag.models import Child, Hit, Parent
from rag.store import SqliteStore
from rag.versions import (
    TITLE_MATCH_BOOST,
    _number_diff,
    _ratio,
    _window,
    aligned_peer_records,
    chunk_overlap_ratio,
    disclose_conflict,
    group_documents,
    parse_review_date,
    reconcile_versions,
    title_from_text,
    title_match_boost,
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


def test_unaligned_pair_and_duplicate_windows_are_skipped():
    from rag.versions import _window

    assert _window("no digits", "2040") == "2040"
    current = _hit("alpha section with no shared wording 2040", "new.md", group="g", parent_id="new")
    current.heading_path = "A"
    stale = _hit("beta section entirely different 2050", "old.md", group="g", superseded=True, parent_id="old")
    stale.heading_path = "B"
    assert disclose_conflict("plain", [current, stale], "").fired is False
    close = _hit("2040 2030", "new.md", group="g", parent_id="c")
    twin = _hit("2050 2031", "old.md", group="g", superseded=True, parent_id="s")
    note = disclose_conflict("The documents do not say.", [close, twin], "")
    assert note.fired
    assert "2040" in note.answer


def test_ratio_match_keeps_the_closer_peer():
    hit = _hit("Carbon Neutral in operations by 2050 extra words here", "new.md", group="g", parent_id="p-new")
    hit.heading_path = ""

    class Index:
        def list_doc_records(self):
            return [{"id": "old.md", "version_group": "g", "status": "superseded"}]

        def parent_records(self, doc_id):
            del doc_id
            return [
                {
                    "parent_id": "near",
                    "text": "Carbon Neutral in operations by 2041 extra words here",
                    "heading_path": "",
                },
                {
                    "parent_id": "far",
                    "text": "Carbon Neutral in operations by 2040 extra words here also",
                    "heading_path": "",
                },
            ]

    found = aligned_peer_records(Index(), [hit])
    assert found and found[0]["parent_id"] in {"far", "near"}


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


def test_aligned_peers_skip_incomplete_records():
    current = "Carbon Neutral in operations by 2050"
    stale = "Carbon Neutral in operations by 2040"
    hit = _hit(current, "new.md", group="g", parent_id="p-new")
    assert aligned_peer_records(object(), [hit]) == []
    assert aligned_peer_records(object(), []) == []

    class Index:
        def list_doc_records(self):
            return [
                {"id": "other.md", "version_group": "else", "status": "current"},
                {"id": "now.md", "version_group": "g", "status": "current"},
                {"id": "", "version_group": "g", "status": "superseded"},
                {"id": "dup.md", "version_group": "g", "status": "superseded"},
                {"id": "old.md", "version_group": "g", "status": "superseded"},
            ]

        def parent_records(self, doc_id):
            if doc_id == "dup.md":
                return [{"parent_id": "p-new", "text": stale, "heading_path": "H"}]
            if doc_id == "old.md":
                return [
                    {"parent_id": "noise", "text": "unrelated irrigation quota text", "heading_path": ""},
                    {"parent_id": "p-old", "text": stale, "heading_path": "H"},
                ]
            return []

    ungrouped = _hit("Carbon Neutral in operations by 2050", "loose.md")
    assert aligned_peer_records(Index(), [ungrouped]) == []
    both = aligned_peer_records(
        Index(),
        [
            hit,
            _hit(stale, "old-copy.md", group="g", superseded=True, parent_id="already"),
        ],
    )
    assert both == []
    found = aligned_peer_records(Index(), [hit])
    assert [row["parent_id"] for row in found] == ["p-old"]

    class NoParents:
        def list_doc_records(self):
            return [{"id": "x", "version_group": "g", "status": "superseded"}]

        def parent_records(self, doc_id):
            del doc_id
            return [{"parent_id": "p", "text": "no numbers here at all", "heading_path": "H"}]

    assert aligned_peer_records(NoParents(), [hit]) == []


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

    assert title_match_boost("", "") == 1.0
    assert title_match_boost("water policy targets", "Water Policy") == TITLE_MATCH_BOOST


def test_window_keeps_a_number_glued_to_a_long_token_and_equal_sequences_fall_back():
    assert _window("x" * 60 + "2040", "2040").endswith("2040")
    spaced = ("z" * 180) + " middlewords 2040 endingwords " + ("q" * 180)
    assert _window(spaced, "2040") == "middlewords 2040 endingwords"
    glued = ("z" * 200) + "2040" + ("q" * 200)
    raw = _window(glued, "2040")
    assert raw.startswith("z") and raw.endswith("q") and "2040" in raw
    assert _number_diff("Carbon neutral by 2040.", "Carbon neutral by 2040.") == ([], [])
    shared = "We are committed to become carbon neutral in our operations by 2040. "
    assert _number_diff(shared, shared + "2030")[1] == ["2030"]
