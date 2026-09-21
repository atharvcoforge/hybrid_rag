import array
import re
import sqlite3
from pathlib import Path

from rag.models import IngestError, make_embed_text

_WORD = re.compile(r"\w+", re.UNICODE)


class Index:
    def __init__(self, path, model_id: str, model_revision: str, pipeline_version: int):
        self.path = Path(path)
        self.model_id = model_id
        self.model_revision = model_revision
        self.pipeline_version = pipeline_version
        self.client = None
        self.db = None
        self.children = None
        self.parents = None

    def open(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path / "side.sqlite")
        self.db.row_factory = sqlite3.Row
        self._schema()
        self._check_saved_identity()
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=str(self.path / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        self.children = _existing(self.client, "children")
        self.parents = _existing(self.client, "parents")
        if self.children is not None:
            _check_collection(self.children, self)

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def matches(self, doc_id, digest, pipeline_version, model_id, revision) -> bool:
        row = self.db.execute("SELECT * FROM files WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return False
        return (
            row["file_sha256"] == digest
            and int(row["pipeline_version"]) == int(pipeline_version)
            and row["model_id"] == model_id
            and row["model_revision"] == revision
        )

    def save_file(self, doc_id, source_path, digest):
        self.db.execute(
            """
            INSERT INTO files (doc_id, source_path, file_sha256, pipeline_version, model_id, model_revision)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_path = excluded.source_path,
                file_sha256 = excluded.file_sha256,
                pipeline_version = excluded.pipeline_version,
                model_id = excluded.model_id,
                model_revision = excluded.model_revision
            """,
            (doc_id, source_path, digest, self.pipeline_version, self.model_id, self.model_revision),
        )
        self.db.commit()

    def cache_get(self, model_id, revision, digest):
        row = self.db.execute(
            "SELECT vector FROM embeddings WHERE model_id = ? AND model_revision = ? AND text_hash = ?",
            (model_id, revision, digest),
        ).fetchone()
        if row is None:
            return None
        values = array.array("f")
        values.frombytes(row["vector"])
        return list(values)

    def cache_put(self, model_id, revision, digest, vector):
        blob = array.array("f", vector).tobytes()
        self.db.execute(
            """
            INSERT INTO embeddings (model_id, model_revision, text_hash, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(model_id, model_revision, text_hash) DO UPDATE SET vector = excluded.vector
            """,
            (model_id, revision, digest, blob),
        )
        self.db.commit()

    def get_tau(self, key: str):
        row = self.db.execute("SELECT tau FROM tau WHERE tau_key = ?", (key,)).fetchone()
        if row is None:
            return None
        return float(row["tau"])

    def set_tau(self, key: str, value: float):
        self.db.execute(
            """
            INSERT INTO tau (tau_key, tau) VALUES (?, ?)
            ON CONFLICT(tau_key) DO UPDATE SET tau = excluded.tau
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
        self._ensure_collections(dim)
        child_ids = [child.chunk_id for child in children]
        child_meta = [_child_meta(child, source) for child in children]
        self._upsert(
            self.children,
            child_ids,
            [child.text for child in children],
            vectors,
            child_meta,
        )
        parent_vector = [1.0] + [0.0] * (dim - 1)
        # Parents are fetched by id. The vector is only here because Chroma requires one.
        self._upsert(
            self.parents,
            [parent.parent_id for parent in parents],
            [parent.text for parent in parents],
            [parent_vector for _parent in parents],
            [_parent_meta(parent, source) for parent in parents],
        )

    def delete_orphans(self, doc_id, keep_child_ids, keep_parent_ids):
        # Delete after the new rows are in. A crash here leaves extras; ingest again removes them.
        _drop_missing(self.children, doc_id, set(keep_child_ids))
        _drop_missing(self.parents, doc_id, set(keep_parent_ids))

    def replace_fts(self, doc_id, children):
        self.db.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
        self.db.executemany(
            "INSERT INTO chunks_fts (chunk_id, doc_id, embed_text) VALUES (?, ?, ?)",
            [(child.chunk_id, doc_id, child.embed_text) for child in children],
        )
        self.db.commit()

    def child_ids(self, doc_id) -> list[str]:
        if self.children is None:
            return []
        found = self.children.get(where={"doc_id": doc_id}, include=["metadatas"])
        return list(found["ids"])

    def dense_search(self, vector, k, doc_id=None):
        if self.children is None or self.children.count() == 0:
            return []
        n = min(k, self.children.count())
        kwargs = {
            "query_embeddings": [list(vector)],
            "n_results": n,
            "include": ["documents", "metadatas"],
        }
        if doc_id:
            kwargs["where"] = {"doc_id": doc_id}
        found = self.children.query(**kwargs)
        if not found["ids"] or not found["ids"][0]:
            return []
        chunks = []
        for chunk_id, body, meta in zip(found["ids"][0], found["documents"][0], found["metadatas"][0]):
            meta = meta or {}
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "parent_id": meta.get("parent_id") or "",
                    "embed_text": make_embed_text(meta.get("heading_path") or "", body or ""),
                }
            )
        return chunks

    def bm25_search(self, text, k, doc_id=None):
        match = fts_query(text)
        if not match or self.children is None:
            return []
        if doc_id:
            rows = self.db.execute(
                """
                SELECT chunk_id FROM chunks_fts
                WHERE chunks_fts MATCH ? AND doc_id = ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (match, doc_id, k),
            ).fetchall()
        else:
            rows = self.db.execute(
                """
                SELECT chunk_id FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (match, k),
            ).fetchall()
        return self._chunks_by_ids([row["chunk_id"] for row in rows])

    def get_parents(self, ids):
        if not ids or self.parents is None:
            return {}
        found = self.parents.get(ids=list(ids), include=["documents", "metadatas"])
        records = {}
        for index, parent_id in enumerate(found["ids"]):
            meta = dict(found["metadatas"][index] or {})
            meta["text"] = found["documents"][index] or ""
            records[parent_id] = meta
        return records

    def _chunks_by_ids(self, ids):
        if not ids:
            return []
        found = self.children.get(ids=list(ids), include=["documents", "metadatas"])
        by_id = {}
        for index, chunk_id in enumerate(found["ids"]):
            meta = found["metadatas"][index] or {}
            body = found["documents"][index] or ""
            by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "parent_id": meta.get("parent_id") or "",
                "embed_text": make_embed_text(meta.get("heading_path") or "", body),
            }
        return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]

    def _ensure_collections(self, dim: int):
        saved = self._meta("dim")
        if saved is not None and int(saved) != dim:
            raise IngestError(
                str(self.path),
                f"index vectors are {saved}-wide and this encoder is {dim}-wide; delete the index directory to rebuild",
            )
        if self.children is not None:
            return
        meta = {
            "hnsw:space": "cosine",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "pipeline_version": int(self.pipeline_version),
            "dim": int(dim),
        }
        self.children = self.client.create_collection(
            name="children",
            metadata=meta,
            embedding_function=_precomputed(),
        )
        self.parents = self.client.create_collection(
            name="parents",
            metadata=meta,
            embedding_function=_precomputed(),
        )
        self._set_meta("model_id", self.model_id)
        self._set_meta("model_revision", self.model_revision)
        self._set_meta("pipeline_version", str(self.pipeline_version))
        self._set_meta("dim", str(dim))

    def _schema(self):
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files (
                doc_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                file_sha256 TEXT NOT NULL,
                pipeline_version INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                embed_text,
                tokenize='unicode61'
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (model_id, model_revision, text_hash)
            );
            CREATE TABLE IF NOT EXISTS tau (
                tau_key TEXT PRIMARY KEY,
                tau REAL NOT NULL
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

    def _meta(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return row["value"]

    def _set_meta(self, key, value):
        self.db.execute(
            """
            INSERT INTO meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self.db.commit()

    def _upsert(self, collection, ids, documents, embeddings, metadatas):
        step = 256
        for start in range(0, len(ids), step):
            collection.upsert(
                ids=ids[start : start + step],
                documents=documents[start : start + step],
                embeddings=embeddings[start : start + step],
                metadatas=metadatas[start : start + step],
            )


def fts_query(text: str) -> str:
    words = _WORD.findall(text)
    if not words:
        return ""
    return " OR ".join('"' + word.replace('"', "") + '"' for word in words)


def _existing(client, name):
    try:
        return client.get_collection(name, embedding_function=_precomputed())
    except Exception:
        return None


def _check_collection(collection, index: Index):
    meta = collection.metadata or {}
    if "model_id" not in meta:
        raise IngestError(str(index.path), "index metadata is missing; delete the index directory to rebuild")
    if (
        meta.get("model_id") != index.model_id
        or meta.get("model_revision") != index.model_revision
        or int(meta.get("pipeline_version")) != int(index.pipeline_version)
    ):
        raise IngestError(
            str(index.path),
            "index was built with "
            f"{meta.get('model_id')}@{meta.get('model_revision')} pipeline {meta.get('pipeline_version')}; "
            "delete the index directory to rebuild",
        )


def _drop_missing(collection, doc_id, keep: set[str]):
    if collection is None:
        return
    found = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    stale = [chunk_id for chunk_id in found["ids"] if chunk_id not in keep]
    if stale:
        collection.delete(ids=stale)


def _child_meta(child, source) -> dict:
    return {
        "doc_id": child.doc_id,
        "parent_id": child.parent_id,
        "file_sha256": source["file_sha256"],
        "source_path": source["source_path"],
        "filename": source["filename"],
        "mime": source["mime"],
        "heading_path": child.heading_path,
        "block_type": child.block_type,
        "start_char": int(child.start_char),
        "end_char": int(child.end_char),
        "page_start": int(child.page_start),
        "page_end": int(child.page_end),
        "child_index": int(child.child_index),
        "parent_index": int(child.parent_index),
        "token_count": int(child.token_count),
        "pipeline_version": int(source["pipeline_version"]),
    }


def _parent_meta(parent, source) -> dict:
    return {
        "doc_id": parent.doc_id,
        "file_sha256": source["file_sha256"],
        "source_path": source["source_path"],
        "filename": source["filename"],
        "mime": source["mime"],
        "heading_path": parent.heading_path,
        "block_type": parent.block_type,
        "start_char": int(parent.start_char),
        "end_char": int(parent.end_char),
        "page_start": int(parent.page_start),
        "page_end": int(parent.page_end),
        "parent_index": int(parent.parent_index),
        "token_count": int(parent.token_count),
        "pipeline_version": int(source["pipeline_version"]),
    }


_PRECOMPUTED = None


def _precomputed():
    global _PRECOMPUTED
    if _PRECOMPUTED is not None:
        return _PRECOMPUTED
    from chromadb.api.types import EmbeddingFunction
    from chromadb.utils.embedding_functions import register_embedding_function

    @register_embedding_function
    class Precomputed(EmbeddingFunction):
        def __init__(self):
            pass

        def __call__(self, input):
            raise RuntimeError("pass embeddings in yourself")

        @staticmethod
        def name() -> str:
            return "precomputed"

        def default_space(self):
            return "cosine"

        def get_config(self):
            return {}

        @staticmethod
        def build_from_config(config):
            return Precomputed()

    _PRECOMPUTED = Precomputed()
    return _PRECOMPUTED
