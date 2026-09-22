"""SQLite-backed vector + FTS store.

One file, one transaction per document. Dense search is exact cosine over an
in-memory float32 matrix — at a few thousand chunks ANN buys nothing here.
"""

from __future__ import annotations

import array
import re
import sqlite3
from pathlib import Path

from rag.models import IngestError

# Quoted compounds force FTS5 adjacency of the unicode61 parts (SKU-7842-XL,
# 14,644), so OR over split tokens cannot rank a distractor that only shares pieces.
_TOKEN = re.compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\w+(?:[-/]\w+)+|\w+",
    re.UNICODE,
)

# Dropped from the OR unless the token carries a digit, hyphen, or slash.
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "with",
        "as",
        "at",
        "from",
        "that",
        "this",
        "it",
        "its",
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "does",
        "do",
        "did",
        "of",
    }
)

class SqliteStore:
    def __init__(self, path, model_id: str, model_revision: str, pipeline_version: int):
        self.path = Path(path)
        self.model_id = model_id
        self.model_revision = model_revision
        self.pipeline_version = pipeline_version
        self.db: sqlite3.Connection | None = None
        self._matrix = None  # numpy float32 (n, dim) or None
        self._chunk_ids: list[str] = []
        self._degraded = False
        self._degraded_reason = ""

    def open(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path / "rag.sqlite"))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self._schema()
        self._check_saved_identity()
        self._load_matrix()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        self._matrix = None
        self._chunk_ids = []

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def matches(self, doc_id, digest, pipeline_version, model_id, revision) -> bool:
        row = self.db.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return False
        return (
            row["file_sha256"] == digest
            and int(row["pipeline_version"]) == int(pipeline_version)
            and row["model_id"] == model_id
            and row["model_revision"] == revision
        )

    def save_file(self, doc_id, source_path, digest):
        # Kept for pipeline compatibility; upsert already writes documents.
        del source_path
        row = self.db.execute(
            "SELECT file_sha256 FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return
        if row["file_sha256"] != digest:
            self.db.execute(
                "UPDATE documents SET file_sha256 = ? WHERE doc_id = ?",
                (digest, doc_id),
            )
            self.db.commit()

    def cache_get(self, model_id, revision, digest):
        row = self.db.execute(
            "SELECT vec FROM embed_cache WHERE model_id = ? AND model_revision = ? AND text_hash = ?",
            (model_id, revision, digest),
        ).fetchone()
        if row is None:
            return None
        values = array.array("f")
        values.frombytes(row["vec"])
        return list(values)

    def cache_put(self, model_id, revision, digest, vector):
        blob = array.array("f", vector).tobytes()
        self.db.execute(
            """
            INSERT INTO embed_cache (model_id, model_revision, text_hash, vec)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(model_id, model_revision, text_hash) DO UPDATE SET vec = excluded.vec
            """,
            (model_id, revision, digest, blob),
        )
        self.db.commit()

    def get_tau(self, key: str):
        row = self.db.execute(
            "SELECT value FROM thresholds WHERE name = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return float(row["value"])

    def set_tau(self, key: str, value: float):
        self.db.execute(
            """
            INSERT INTO thresholds (name, value, fitted_at, fitted_on_n, holdout_metric)
            VALUES (?, ?, datetime('now'), 0, NULL)
            ON CONFLICT(name) DO UPDATE SET
                value = excluded.value,
                fitted_at = excluded.fitted_at
            """,
            (key, float(value)),
        )
        self.db.commit()

    def upsert(self, children, parents, vectors, source: dict):
        if len(children) != len(vectors):
            raise IngestError(source["source_path"], "embedding count does not match chunks")
        if not children:
            raise IngestError(source["source_path"], "no chunks")
        dim = len(vectors[0])
        if any(len(vector) != dim for vector in vectors):
            raise IngestError(source["source_path"], "embedding width changed inside one file")
        self._ensure_dim(dim)
        doc_id = children[0].doc_id
        self.db.execute("BEGIN")
        try:
            self.db.execute(
                """
                INSERT INTO documents (
                    doc_id, source_path, filename, mime, file_sha256,
                    pipeline_version, model_id, model_revision, page_count, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    filename = excluded.filename,
                    mime = excluded.mime,
                    file_sha256 = excluded.file_sha256,
                    pipeline_version = excluded.pipeline_version,
                    model_id = excluded.model_id,
                    model_revision = excluded.model_revision,
                    page_count = excluded.page_count,
                    ingested_at = excluded.ingested_at
                """,
                (
                    doc_id,
                    source["source_path"],
                    source["filename"],
                    source["mime"],
                    source["file_sha256"],
                    int(source["pipeline_version"]),
                    self.model_id,
                    self.model_revision,
                    max((child.page_end for child in children), default=0),
                ),
            )
            self.db.execute("DELETE FROM children_fts WHERE doc_id = ?", (doc_id,))
            self.db.execute(
                "DELETE FROM vectors WHERE chunk_id IN (SELECT chunk_id FROM children WHERE doc_id = ?)",
                (doc_id,),
            )
            self.db.execute("DELETE FROM children WHERE doc_id = ?", (doc_id,))
            self.db.execute("DELETE FROM parents WHERE doc_id = ?", (doc_id,))
            self.db.executemany(
                """
                INSERT INTO parents (
                    parent_id, doc_id, text, heading_path, block_type,
                    page_start, page_end, norm_start, norm_end, parent_index, token_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        parent.parent_id,
                        parent.doc_id,
                        parent.text,
                        parent.heading_path,
                        parent.block_type,
                        int(parent.page_start),
                        int(parent.page_end),
                        int(parent.start_char),
                        int(parent.end_char),
                        int(parent.parent_index),
                        int(parent.token_count),
                    )
                    for parent in parents
                ],
            )
            self.db.executemany(
                """
                INSERT INTO children (
                    chunk_id, parent_id, doc_id, text, embed_text, context_prefix,
                    heading_path, block_type, page_start, page_end,
                    norm_start, norm_end, bbox_json, child_index, token_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        child.chunk_id,
                        child.parent_id,
                        child.doc_id,
                        child.text,
                        child.embed_text,
                        getattr(child, "context_prefix", "") or "",
                        child.heading_path,
                        child.block_type,
                        int(child.page_start),
                        int(child.page_end),
                        int(child.start_char),
                        int(child.end_char),
                        None,
                        int(child.child_index),
                        int(child.token_count),
                    )
                    for child in children
                ],
            )
            self.db.executemany(
                """
                INSERT INTO vectors (chunk_id, model_id, model_revision, dim, vec)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        child.chunk_id,
                        self.model_id,
                        self.model_revision,
                        dim,
                        array.array("f", vector).tobytes(),
                    )
                    for child, vector in zip(children, vectors)
                ],
            )
            fts_rows = []
            for child in children:
                ident, prose = _split_fts(child.embed_text)
                fts_rows.append((child.chunk_id, child.doc_id, ident, prose))
            self.db.executemany(
                "INSERT INTO children_fts (chunk_id, doc_id, ident, prose) VALUES (?, ?, ?, ?)",
                fts_rows,
            )
            gen = int(self._meta("index_generation") or "0") + 1
            self._set_meta("index_generation", str(gen), commit=False)
            self._set_meta("model_id", self.model_id, commit=False)
            self._set_meta("model_revision", self.model_revision, commit=False)
            self._set_meta("pipeline_version", str(self.pipeline_version), commit=False)
            self._set_meta("dim", str(dim), commit=False)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self._load_matrix()

    def delete_orphans(self, doc_id, keep_child_ids, keep_parent_ids):
        keep_c = set(keep_child_ids)
        keep_p = set(keep_parent_ids)
        child_rows = self.db.execute(
            "SELECT chunk_id FROM children WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        stale_c = [row["chunk_id"] for row in child_rows if row["chunk_id"] not in keep_c]
        parent_rows = self.db.execute(
            "SELECT parent_id FROM parents WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        stale_p = [row["parent_id"] for row in parent_rows if row["parent_id"] not in keep_p]
        for chunk_id in stale_c:
            self.db.execute("DELETE FROM vectors WHERE chunk_id = ?", (chunk_id,))
            self.db.execute("DELETE FROM children_fts WHERE chunk_id = ?", (chunk_id,))
            self.db.execute("DELETE FROM children WHERE chunk_id = ?", (chunk_id,))
        for parent_id in stale_p:
            self.db.execute("DELETE FROM parents WHERE parent_id = ?", (parent_id,))
        self.db.commit()

    def replace_fts(self, doc_id, children):
        # FTS is written inside upsert; kept as a no-op for pipeline compatibility.
        del doc_id, children

    def child_ids(self, doc_id) -> list[str]:
        rows = self.db.execute(
            "SELECT chunk_id FROM children WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        return [row["chunk_id"] for row in rows]

    def list_docs(self) -> list[str]:
        rows = self.db.execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall()
        return [row["doc_id"] for row in rows]

    def purge_doc(self, doc_id: str) -> None:
        self.db.execute("DELETE FROM children_fts WHERE doc_id = ?", (doc_id,))
        self.db.execute(
            "DELETE FROM vectors WHERE chunk_id IN (SELECT chunk_id FROM children WHERE doc_id = ?)",
            (doc_id,),
        )
        self.db.execute("DELETE FROM children WHERE doc_id = ?", (doc_id,))
        self.db.execute("DELETE FROM parents WHERE doc_id = ?", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self.db.commit()
        self._load_matrix()

    def purge_missing(self, keep_doc_ids) -> list[str]:
        keep = set(keep_doc_ids)
        removed = []
        for doc_id in self.list_docs():
            if doc_id not in keep:
                self.purge_doc(doc_id)
                removed.append(doc_id)
        return removed

    def dense_search(self, vector, k, doc_id=None):
        if self._matrix is None or len(self._chunk_ids) == 0:
            return []
        import numpy as np

        query = np.asarray(vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self._matrix.shape[1]:
            raise IngestError(str(self.path), "query vector width does not match the index")
        # Vectors are L2-normalised at ingest; cosine == dot product.
        scores = self._matrix @ query
        if doc_id:
            allowed = {
                row["chunk_id"]
                for row in self.db.execute(
                    "SELECT chunk_id FROM children WHERE doc_id = ?", (doc_id,)
                )
            }
            mask = np.array([chunk_id in allowed for chunk_id in self._chunk_ids], dtype=bool)
            if not mask.any():
                return []
            scores = np.where(mask, scores, -np.inf)
        n = min(k, int(np.isfinite(scores).sum()))
        if n <= 0:
            return []
        if n >= len(scores):
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, n - 1)[:n]
            order = part[np.argsort(-scores[part])]
        out = []
        for index in order:
            chunk_id = self._chunk_ids[int(index)]
            row = self.db.execute(
                "SELECT chunk_id, parent_id, embed_text FROM children WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                continue
            out.append(
                {
                    "chunk_id": row["chunk_id"],
                    "parent_id": row["parent_id"],
                    "embed_text": row["embed_text"],
                    "dense_score": float(scores[int(index)]),
                }
            )
        return out

    def bm25_search(self, text, k, doc_id=None):
        match = fts_query(text)
        if not match:
            return []
        # Weighted: ident column full weight, prose at 0.4.
        if doc_id:
            rows = self.db.execute(
                """
                SELECT chunk_id, bm25(children_fts, 1.0, 0.4) AS score
                FROM children_fts
                WHERE children_fts MATCH ? AND doc_id = ?
                ORDER BY score
                LIMIT ?
                """,
                (match, doc_id, k),
            ).fetchall()
        else:
            rows = self.db.execute(
                """
                SELECT chunk_id, bm25(children_fts, 1.0, 0.4) AS score
                FROM children_fts
                WHERE children_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match, k),
            ).fetchall()
        out = []
        for row in rows:
            child = self.db.execute(
                "SELECT chunk_id, parent_id, embed_text FROM children WHERE chunk_id = ?",
                (row["chunk_id"],),
            ).fetchone()
            if child is None:
                continue
            out.append(
                {
                    "chunk_id": child["chunk_id"],
                    "parent_id": child["parent_id"],
                    "embed_text": child["embed_text"],
                    # bm25() is lower-is-better; flip so callers see higher-is-better.
                    "bm25_score": float(-row["score"]),
                }
            )
        return out

    def get_parents(self, ids):
        if not ids:
            return {}
        records = {}
        for parent_id in ids:
            row = self.db.execute(
                """
                SELECT p.*, d.source_path, d.file_sha256, d.filename, d.mime
                FROM parents p
                JOIN documents d ON d.doc_id = p.doc_id
                WHERE p.parent_id = ?
                """,
                (parent_id,),
            ).fetchone()
            if row is None:
                continue
            records[parent_id] = {
                "text": row["text"],
                "heading_path": row["heading_path"] or "",
                "source_path": row["source_path"] or "",
                "file_sha256": row["file_sha256"] or "",
                "filename": row["filename"] or "",
                "mime": row["mime"] or "",
                "block_type": row["block_type"] or "",
                "start_char": int(row["norm_start"] or 0),
                "end_char": int(row["norm_end"] or 0),
                "norm_start": int(row["norm_start"] or 0),
                "norm_end": int(row["norm_end"] or 0),
                "page_start": int(row["page_start"] or 0),
                "page_end": int(row["page_end"] or 0),
                "parent_index": int(row["parent_index"] or 0),
                "token_count": int(row["token_count"] or 0),
            }
        return records

    def index_generation(self) -> int:
        return int(self._meta("index_generation") or "0")

    def _ensure_dim(self, dim: int):
        saved = self._meta("dim")
        if saved is not None and int(saved) != dim:
            raise IngestError(
                str(self.path),
                f"index vectors are {saved}-wide and this encoder is {dim}-wide; delete the index directory to rebuild",
            )

    def _schema(self):
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
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
            );
            CREATE TABLE IF NOT EXISTS parents (
                parent_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                text TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                block_type TEXT NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                norm_start INTEGER NOT NULL,
                norm_end INTEGER NOT NULL,
                parent_index INTEGER NOT NULL,
                token_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS children (
                chunk_id TEXT PRIMARY KEY,
                parent_id TEXT NOT NULL REFERENCES parents(parent_id),
                doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                text TEXT NOT NULL,
                embed_text TEXT NOT NULL,
                context_prefix TEXT NOT NULL DEFAULT '',
                heading_path TEXT NOT NULL,
                block_type TEXT NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                norm_start INTEGER NOT NULL,
                norm_end INTEGER NOT NULL,
                bbox_json TEXT,
                child_index INTEGER NOT NULL,
                token_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id TEXT PRIMARY KEY REFERENCES children(chunk_id),
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embed_cache (
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vec BLOB NOT NULL,
                PRIMARY KEY (model_id, model_revision, text_hash)
            );
            CREATE TABLE IF NOT EXISTS thresholds (
                name TEXT PRIMARY KEY,
                value REAL NOT NULL,
                fitted_at TEXT,
                fitted_on_n INTEGER,
                holdout_metric REAL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS children_fts USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                ident,
                prose,
                tokenize='unicode61'
            );
            """
        )

    def _check_saved_identity(self):
        saved = self._meta("model_id")
        if saved is None:
            return
        revision = self._meta("model_revision")
        version = self._meta("pipeline_version")
        if saved != self.model_id or revision != self.model_revision or int(version) != int(self.pipeline_version):
            raise IngestError(
                str(self.path),
                f"index was built with {saved}@{revision} pipeline {version}; delete the index directory to rebuild",
            )

    def _load_matrix(self):
        rows = self.db.execute(
            """
            SELECT v.chunk_id, v.dim, v.vec
            FROM vectors v
            JOIN children c ON c.chunk_id = v.chunk_id
            ORDER BY c.child_index, c.chunk_id
            """
        ).fetchall()
        if not rows:
            self._matrix = None
            self._chunk_ids = []
            return
        import numpy as np

        dim = int(rows[0]["dim"])
        ids = []
        vecs = []
        for row in rows:
            values = array.array("f")
            values.frombytes(row["vec"])
            if len(values) != dim:
                self._degraded = True
                self._degraded_reason = "vector width mismatch"
                continue
            vecs.append(values)
            ids.append(row["chunk_id"])
        if not ids:
            self._matrix = None
            self._chunk_ids = []
            return
        self._matrix = np.asarray(vecs, dtype=np.float32)
        self._chunk_ids = ids

    def _meta(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return row["value"]

    def _set_meta(self, key, value, commit=True):
        self.db.execute(
            """
            INSERT INTO meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        if commit:
            self.db.commit()


def fts_query(text: str) -> str:
    words = _TOKEN.findall(text)
    kept = []
    for word in words:
        lower = word.casefold()
        if lower in _STOP and not re.search(r"[\d\-/]", word):
            continue
        kept.append(word)
    if not kept:
        # Fall back to the raw tokens so an all-stopword query still runs.
        kept = words
    if not kept:
        return ""
    return " OR ".join('"' + word.replace('"', "") + '"' for word in kept)


def _split_fts(embed_text: str) -> tuple[str, str]:
    tokens = _TOKEN.findall(embed_text)
    ident = []
    prose = []
    for token in tokens:
        if re.search(r"[\d\-/]", token) or "," in token:
            ident.append(token)
        else:
            prose.append(token)
    return " ".join(ident), " ".join(prose)


# Public name used throughout the codebase.
Index = SqliteStore
