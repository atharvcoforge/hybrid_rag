import array
import re
import sqlite3
from pathlib import Path

import pytest

from rag.models import Child, IngestError, Parent
from rag.store import SqliteStore, _split_fts, fts_query, repair_extracted_text

V1 = [1.0, 0.0, 0.0, 0.0]
V2 = [0.0, 1.0, 0.0, 0.0]


def _source(doc_id: str, digest: str = "sha-a") -> dict[str, object]:
    return {
        "source_path": doc_id,
        "filename": doc_id,
        "mime": "text/markdown",
        "file_sha256": digest,
        "pipeline_version": 3,
    }


def _parent(
    doc_id: str,
    parent_id: str,
    text: str,
    *,
    index: int = 0,
    derived: bool = False,
    heading: str = "",
    block: str = "prose",
    start: int = 0,
    end: int = 0,
    page: int = 0,
    tokens: int = 0,
) -> Parent:
    return Parent(
        parent_id,
        doc_id,
        text,
        heading,
        block,
        start,
        end,
        page,
        page,
        index,
        tokens,
        derived,
    )


def _child(
    doc_id: str,
    parent_id: str,
    chunk_id: str,
    text: str,
    *,
    index: int = 0,
    prefix: str = "",
) -> Child:
    child = Child(
        chunk_id,
        parent_id,
        doc_id,
        text,
        text,
        "",
        "prose",
        0,
        len(text),
        0,
        0,
        index,
        index,
        1,
    )
    child.context_prefix = prefix
    return child


def test_repair_and_fts_query_edges() -> None:
    assert repair_extracted_text("") == ""
    assert repair_extracted_text("hello") == "hello"
    untouched = "© 2020 other"
    assert repair_extracted_text(untouched) == untouched
    fixed = repair_extracted_text("pre © ©20 2260 C2o6fo Crgoef orge post")
    assert "© 2026 Coforge" in fixed

    assert fts_query("") == ""
    assert fts_query("???") == ""
    assert fts_query("the and of") == '"the" OR "and" OR "of"'
    assert "greenhouse" in fts_query("ghg")
    assert "greenhouse" in fts_query("ghg greenhouse")
    assert "weee" in fts_query("e-waste")
    assert "weee" in fts_query("ewaste")
    titled = fts_query("who signed")
    assert '"president"' in titled
    assert '"executive"' in titled
    assert '"director"' in titled
    already = fts_query("signed by the president executive director")
    assert already.count('"president"') == 1
    assert "president" not in fts_query("blue housing")

    ident, prose = _split_fts("plain SKU-7842-XL 14,644 06/21")
    assert "SKU-7842-XL" in ident
    assert "14,644" in ident
    assert "06/21" in ident
    assert prose == "plain"


def test_closed_store_and_old_schema_migration(tmp_path: Path) -> None:
    closed = SqliteStore(tmp_path / "closed", "m", "r", 3)
    closed.close()
    with pytest.raises(RuntimeError, match="store is not open"):
        closed.list_docs()
    with pytest.raises(IngestError, match="index is not open"):
        closed.integrity_problems()

    old = tmp_path / "old"
    old.mkdir()
    db = sqlite3.connect(old / "rag.sqlite")
    db.execute(
        """
        CREATE TABLE parents (
            parent_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            heading_path TEXT NOT NULL,
            block_type TEXT NOT NULL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            norm_start INTEGER NOT NULL,
            norm_end INTEGER NOT NULL,
            parent_index INTEGER NOT NULL,
            token_count INTEGER NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            filename TEXT NOT NULL,
            mime TEXT NOT NULL,
            file_sha256 TEXT NOT NULL,
            pipeline_version INTEGER NOT NULL,
            model_id TEXT NOT NULL,
            model_revision TEXT NOT NULL,
            page_count INTEGER NOT NULL DEFAULT 0,
            ingested_at TEXT NOT NULL
        )
        """
    )
    db.commit()
    db.close()
    with SqliteStore(old, "m", "r", 3) as store:
        parent_cols = {row[1] for row in store._db().execute("PRAGMA table_info(parents)")}
        doc_cols = {row[1] for row in store._db().execute("PRAGMA table_info(documents)")}
    assert "derived" in parent_cols
    assert "review_date" in doc_cols
    assert "superseded_by" in doc_cols


def test_store_runtime_branches(tmp_path: Path) -> None:
    path = tmp_path / "index"
    store = SqliteStore(path, "model-a", "rev-a", 3)
    with store as opened:
        assert opened is store
        assert store.dense_search(V1, 5) == []
        store._matrix = []
        store._chunk_ids = []
        assert store.dense_search(V1, 5) == []
        assert store.list_doc_records() == []
        assert store.purge_missing([]) == []
        assert store.index_generation() == 0
        assert store.integrity_problems() == []
        assert store.get_parents([]) == {}
        assert store.get_tau("missing") is None
        assert store.cache_get("m", "r", "missing") is None
        store.save_file("missing", "path", "sha")

        with pytest.raises(IngestError, match="embedding count"):
            store.upsert([_child("a.md", "p", "c", "text")], [], [], _source("a.md"))
        with pytest.raises(IngestError, match="no chunks"):
            store.upsert([], [], [], _source("a.md"))
        with pytest.raises(IngestError, match="embedding width"):
            store.upsert(
                [_child("a.md", "p1", "c1", "a"), _child("a.md", "p2", "c2", "b")],
                [_parent("a.md", "p1", "a"), _parent("a.md", "p2", "b")],
                [[1.0, 0.0], [1.0]],
                _source("a.md"),
            )
        duplicate = _parent("bad.md", "same", "x")
        with pytest.raises(sqlite3.IntegrityError):
            store.upsert(
                [_child("bad.md", "same", "c", "text")],
                [duplicate, duplicate],
                [V1],
                _source("bad.md"),
            )
        assert store.list_docs() == []

        copyright_text = "footer © ©20 2260 C2o6fo Crgoef orge end"
        parents = [
            _parent("a.md", "p0", "alpha derived", index=0, derived=True, tokens=2),
            _parent("a.md", "p1", "", index=1),
            _parent("a.md", "p2", "no match here", index=2, heading="Section", block="prose", start=4, end=9, page=2, tokens=5),
            _parent("a.md", "p3", "alpha bare", index=3),
            _parent("a.md", "p4", "beta hit", index=4, heading="H", start=3, end=6, page=1, tokens=2),
            _parent("a.md", "p5", copyright_text, index=5),
        ]
        children = [
            _child("a.md", "p0", "c0", "alpha plain SKU-7842-XL", index=0),
            _child("a.md", "p2", "c2", "beta plain 14,644 06/21", index=1, prefix="ctx"),
            _child("a.md", "p4", "c4", "beta hit", index=2),
            _child("a.md", "p5", "c5", "copyright line", index=3),
        ]
        store.upsert(children, parents, [V1, V2, V1, V2], _source("a.md"))
        blank = _source("b.md", digest="")
        blank["filename"] = ""
        blank["source_path"] = ""
        blank["mime"] = ""
        store.upsert(
            [_child("b.md", "pb", "cb", "other token", prefix="")],
            [_parent("b.md", "pb", "other token", heading="", block="")],
            [V2],
            blank,
        )
        store._db().execute(
            """
            UPDATE documents
            SET title = ?, status = ?, review_date = ?, version_group = ?,
                supersedes = ?, superseded_by = ?
            WHERE doc_id = ?
            """,
            ("Title", "superseded", "2024-01-01", "grp", "old.md", "new.md", "a.md"),
        )
        store._db().commit()
        store.replace_fts("a.md", children)
        store.save_file("a.md", "a.md", "sha-a")
        store.save_file("a.md", "a.md", "sha-b")
        assert store.matches("a.md", "sha-b", 3, "model-a", "rev-a")
        assert not store.matches("missing", "sha-b", 3, "model-a", "rev-a")
        assert not store.matches("a.md", "nope", 3, "model-a", "rev-a")
        assert not store.matches("a.md", "sha-b", 2, "model-a", "rev-a")
        assert not store.matches("a.md", "sha-b", 3, "model-b", "rev-a")
        assert not store.matches("a.md", "sha-b", 3, "model-a", "rev-b")

        store.cache_put("m", "r", "h", [0.5, 0.25])
        store.cache_put("m", "r", "h", [0.5, 0.25])
        cached = store.cache_get("m", "r", "h")
        assert cached == pytest.approx([0.5, 0.25])
        store.set_tau("k", 0.5)
        store.set_tau("k", 0.95)
        assert store.get_tau("k") == pytest.approx(0.95)
        assert store.index_generation() >= 1

        listed = {row["id"]: row for row in store.list_doc_records()}
        assert listed["a.md"]["filename"] == "a.md"
        assert listed["a.md"]["title"] == "Title"
        assert listed["a.md"]["status"] == "superseded"
        assert listed["b.md"]["filename"] == "b.md"
        assert listed["b.md"]["title"] == "b.md"
        assert listed["b.md"]["source_path"] == "b.md"
        assert listed["b.md"]["status"] == "current"

        records = store.get_parents(["p5", "p1", "pb", "missing"])
        assert "missing" not in records
        assert "© 2026 Coforge" in records["p5"]["text"]
        assert records["p5"]["status"] == "superseded"
        assert records["p5"]["superseded"] is True
        assert records["p1"]["text"] == ""
        assert records["p1"]["heading_path"] == ""
        assert records["pb"]["status"] == "current"
        assert records["pb"]["superseded"] is False
        assert records["pb"]["mime"] == ""
        assert records["pb"]["file_sha256"] == ""

        alpha = re.compile("alpha")
        bare = store.first_parent_matching("a.md", alpha)
        assert bare is not None
        assert bare["parent_id"] == "p3"
        assert bare["child_id"] == "p3"
        beta = store.first_parent_matching("a.md", re.compile("beta hit"))
        assert beta is not None
        assert beta["child_id"] == "c4"
        assert store.first_parent_matching("a.md", re.compile("zzz")) is None

        store._db().execute(
            """
            INSERT INTO documents (
                doc_id, source_path, filename, mime, file_sha256,
                pipeline_version, model_id, model_revision, page_count, ingested_at
            ) VALUES ('empty-doc', 'empty-doc', 'empty-doc', 'text/plain', 'x', 3,
                      'model-a', 'rev-a', 0, datetime('now'))
            """
        )
        store._db().execute(
            """
            INSERT INTO parents (
                parent_id, doc_id, text, heading_path, block_type,
                page_start, page_end, norm_start, norm_end, parent_index, token_count, derived
            ) VALUES ('ghost', 'missing-doc', 'alpha', '', 'prose', 0, 0, 0, 0, 0, 1, 0)
            """
        )
        store._db().execute(
            "INSERT INTO children_fts (chunk_id, doc_id, ident, prose) VALUES ('ghost-fts', 'z', 'x', 'y')"
        )
        store._db().commit()
        assert store.first_parent_matching("empty-doc", alpha) is None
        assert store.first_parent_matching("missing-doc", alpha) is None
        assert store.integrity_problems()

        assert store.child_ids("a.md")
        keep_children = store.child_ids("a.md")
        keep_parents = ["p0", "p1", "p2", "p3", "p4", "p5"]
        store.delete_orphans("a.md", keep_children, keep_parents)
        store.upsert(
            [
                _child("c.md", "pc1", "cc1", "keep me"),
                _child("c.md", "pc2", "cc2", "drop me"),
            ],
            [
                _parent("c.md", "pc1", "keep me"),
                _parent("c.md", "pc2", "drop me", index=1),
            ],
            [V1, V2],
            _source("c.md"),
        )
        store.delete_orphans("c.md", ["cc1"], ["pc1"])
        assert store.child_ids("c.md") == ["cc1"]
        assert "pc2" not in store.get_parents(["pc2"])

        assert store.dense_search(V1, 0) == []
        with pytest.raises(IngestError, match="query vector width"):
            store.dense_search([1.0, 0.0], 3)
        with pytest.raises(IngestError, match="query vector width"):
            store.dense_search([[1.0, 0.0, 0.0, 0.0]], 3)
        assert store.dense_search(V1, 5, doc_id="missing.md") == []
        filtered = store.dense_search(V1, 1, doc_id="a.md")
        assert filtered
        ranked = store.dense_search(V1, 5)
        assert len(ranked) >= 2
        assert store.bm25_search("", 5) == []
        assert store.bm25_search("alpha", 5, doc_id="a.md")
        assert store.bm25_search("alpha", 5, doc_id="missing.md") == []
        assert store.bm25_search("plain", 5)

        store._db().execute("DELETE FROM children WHERE chunk_id = ?", ("c2",))
        store._db().commit()
        dense_after = store.dense_search(V1, 5)
        assert "c2" not in {hit["chunk_id"] for hit in dense_after}
        lexical_after = store.bm25_search("plain", 5)
        assert "c2" not in {hit["chunk_id"] for hit in lexical_after}
        assert lexical_after

        blob = array.array("f", [1.0]).tobytes()
        store._db().execute("UPDATE vectors SET vec = ?", (blob,))
        store._db().commit()
        store._load_matrix()
        assert store._matrix is None
        assert store._chunk_ids == []
        assert store._degraded is True
        assert store._degraded_reason == "vector width mismatch"

        removed = store.purge_missing(["a.md"])
        assert "b.md" in removed
        assert "a.md" not in removed

        with pytest.raises(IngestError, match="delete the index directory to rebuild"):
            store.upsert(
                [_child("wide.md", "pw", "cw", "wide")],
                [_parent("wide.md", "pw", "wide")],
                [[1.0, 0.0]],
                _source("wide.md"),
            )

    store.close()
    with pytest.raises(RuntimeError, match="store is not open"):
        store.get_tau("k")


def test_saved_identity_mismatches(tmp_path: Path) -> None:
    path = tmp_path / "index"
    with SqliteStore(path, "model-a", "rev-a", 3) as store:
        store.upsert(
            [_child("a.md", "p", "c", "hello")],
            [_parent("a.md", "p", "hello")],
            [V1],
            _source("a.md"),
        )
        store._set_meta("extra", "1", commit=False)
        store._db().commit()

    with SqliteStore(path, "model-a", "rev-a", 3) as store:
        assert store.list_docs() == ["a.md"]

    wrong_model = SqliteStore(path, "model-b", "rev-a", 3)
    with pytest.raises(IngestError, match="delete the index directory to rebuild") as model_exc:
        wrong_model.open()
    assert "pipeline version changed" not in str(model_exc.value)
    wrong_model.close()

    wrong_rev = SqliteStore(path, "model-a", "rev-b", 3)
    with pytest.raises(IngestError, match="delete the index directory to rebuild") as rev_exc:
        wrong_rev.open()
    assert "pipeline version changed" not in str(rev_exc.value)
    wrong_rev.close()

    wrong_version = SqliteStore(path, "model-a", "rev-a", 9)
    with pytest.raises(IngestError, match="pipeline version changed from 3 to 9") as version_exc:
        wrong_version.open()
    assert "delete the index directory to rebuild" in str(version_exc.value)
    wrong_version.close()

    with SqliteStore(path, "model-a", "rev-a", 3) as store:
        store._db().execute("DELETE FROM meta WHERE key = 'pipeline_version'")
        store._db().commit()
    missing_version = SqliteStore(path, "model-a", "rev-a", 3)
    with pytest.raises(IngestError, match="pipeline version changed from None to 3"):
        missing_version.open()
    missing_version.close()
