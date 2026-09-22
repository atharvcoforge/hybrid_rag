import hashlib
from pathlib import Path

from rag.chunk import chunk_document
from rag.embed import embed_texts
from rag.models import EMBED_MODEL, EMBED_REVISION, MAX_CHUNKS, MAX_FILE_BYTES, PIPELINE_VERSION, IngestError, Ingested
from rag.parse import MIMES, parse_file
from rag.retrieve import retrieve
from rag.store import Index


def ingest(
    path,
    index_dir,
    encode=None,
    count_tokens=None,
    model_id=None,
    model_revision=None,
    *,
    contextual: bool = False,
    context_fn=None,
) -> list[Ingested]:
    path = Path(path)
    index_dir = Path(index_dir)
    model_id = model_id or EMBED_MODEL
    model_revision = model_revision or EMBED_REVISION
    if encode is None:
        from rag.embed import encode_documents

        encode = encode_documents
    if count_tokens is None:
        from rag.embed import count_tokens as count_tokens
    if not path.exists():
        raise IngestError(str(path), "file not found")
    if path.is_file() and path.suffix.lower() not in MIMES:
        raise IngestError(str(path), "unsupported file type")
    index = Index(index_dir, model_id, model_revision, PIPELINE_VERSION)
    index.open()
    try:
        files = _files(path, index_dir)
        results = [
            _ingest_one(
                file,
                root,
                index,
                encode,
                count_tokens,
                model_id,
                model_revision,
                contextual=contextual,
                context_fn=context_fn,
            )
            for file, root in files
        ]
        if path.is_dir():
            seen = {item.doc_id for item in results}
            for doc_id in index.purge_missing(seen):
                results.append(Ingested(doc_id, "purged", 0))
        return results
    finally:
        index.close()


def query(text, index_dir, doc_id=None, embed_query=None, rerank=None, model_id=None, model_revision=None, mode="cascade"):
    from rag.models import QueryError

    if not text or not text.strip():
        raise QueryError("empty query")
    model_id = model_id or EMBED_MODEL
    model_revision = model_revision or EMBED_REVISION
    if embed_query is None:
        from rag.embed import encode_query as embed_query
    if rerank is None:
        from rag.embed import rerank_scores as rerank
    index = Index(Path(index_dir), model_id, model_revision, PIPELINE_VERSION)
    index.open()
    try:
        return retrieve(index, text, embed_query, rerank=rerank, doc_id=doc_id, mode=mode)
    finally:
        index.close()


def _ingest_one(
    file,
    root,
    index,
    encode,
    count_tokens,
    model_id,
    model_revision,
    *,
    contextual: bool = False,
    context_fn=None,
) -> Ingested:
    resolved = _inside(root, file)
    if resolved.stat().st_size > MAX_FILE_BYTES:
        raise IngestError(str(file), "file exceeds 50 MB")
    digest = _sha256(resolved)
    doc_id = resolved.relative_to(root.resolve()).as_posix()
    if index.matches(doc_id, digest, PIPELINE_VERSION, model_id, model_revision):
        return Ingested(doc_id, "skipped", 0)
    mime, blocks = parse_file(resolved)
    warnings = []
    for block in blocks:
        if block.flagged:
            warnings.append(f"flagged {block.kind} on page {block.page}")
        if block.text.startswith("[page error]"):
            warnings.append(block.text)
    ctx = context_fn if contextual else None
    _parents, children = chunk_document(doc_id, blocks, count_tokens, context_fn=ctx)
    if not children:
        raise IngestError(str(file), "no chunks")
    if len(children) > MAX_CHUNKS:
        raise IngestError(str(file), "parser produced too many chunks")
    vectors = embed_texts(
        [child.embed_text for child in children],
        encode,
        cache_get=index.cache_get,
        cache_put=index.cache_put,
        model_id=model_id,
        revision=model_revision,
    )
    source = {
        "source_path": doc_id,
        "filename": resolved.name,
        "mime": mime,
        "file_sha256": digest,
        "pipeline_version": PIPELINE_VERSION,
    }
    index.upsert(children, _parents, vectors, source)
    index.delete_orphans(doc_id, [child.chunk_id for child in children], [parent.parent_id for parent in _parents])
    index.replace_fts(doc_id, children)
    index.save_file(doc_id, doc_id, digest)
    return Ingested(doc_id, "indexed", len(children), warnings=warnings or None)


def _files(path: Path, index_dir: Path) -> list[tuple[Path, Path]]:
    if path.is_dir():
        root = path
        found = []
        for file in sorted(root.rglob("*")):
            if not file.is_file() or file.name.startswith("."):
                continue
            if file.suffix.lower() not in MIMES:
                continue
            if _is_under(index_dir, file):
                continue
            found.append(file)
        return [(file, root) for file in found]
    if path.suffix.lower() not in MIMES:
        raise IngestError(str(path), "unsupported file type")
    return [(path, path.parent)]


def _inside(root: Path, file: Path) -> Path:
    root_resolved = root.resolve()
    try:
        resolved = file.resolve(strict=True)
    except OSError as exc:
        raise IngestError(str(file), "unreadable file") from exc
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise IngestError(str(file), "path escapes the ingest folder")
    return resolved


def _is_under(root: Path, file: Path) -> bool:
    try:
        file.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()
