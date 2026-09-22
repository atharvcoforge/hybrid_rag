import sqlite3

from rag.store import fts_query


def test_fts_query_keeps_hyphenated_id_as_one_term():
    assert fts_query("SKU-7842-XL") == '"SKU-7842-XL"'


def test_fts_query_keeps_comma_number_as_one_term():
    assert fts_query("14,644") == '"14,644"'


def test_fts_query_keeps_slash_code_and_surrounding_words():
    assert fts_query("PPN 06/21") == '"PPN" OR "06/21"'


def test_fts_query_still_ors_plain_words():
    assert fts_query("blue housing") == '"blue" OR "housing"'


def test_fts_query_empty_when_no_tokens():
    assert fts_query("???") == ""


def test_compound_query_does_not_match_split_distractor():
    # Old OR-of-parts matched the distractor; the compound phrase must not.
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(id UNINDEXED, body, tokenize='unicode61')")
    db.executemany(
        "INSERT INTO t(id, body) VALUES (?, ?)",
        [
            ("hit", "India baseline total emissions 14,644 tCO2e"),
            ("miss", "room 14 on floor 644 elsewhere"),
            ("sku", "part SKU-7842-XL nests on the rail"),
            ("sku_miss", "SKU sticker and XL size chart"),
        ],
    )
    number_ids = [
        row[0]
        for row in db.execute(
            "SELECT id FROM t WHERE t MATCH ? ORDER BY bm25(t)",
            (fts_query("14,644"),),
        )
    ]
    sku_ids = [
        row[0]
        for row in db.execute(
            "SELECT id FROM t WHERE t MATCH ? ORDER BY bm25(t)",
            (fts_query("SKU-7842-XL"),),
        )
    ]
    assert number_ids == ["hit"]
    assert sku_ids == ["sku"]
