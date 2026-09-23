import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from rag.chunk import chunk_document
from rag.embed import embed_texts
from rag.models import (
    EMBED_MODEL,
    EMBED_REVISION,
    MAX_CHUNKS,
    MAX_FILE_BYTES,
    PIPELINE_VERSION,
    Ingested,
    IngestError,
    Retrieval,
)
from rag.parse import MIMES, parse_file
from rag.retrieve import retrieve
from rag.store import Index, SqliteStore


def _open_for_ingest(index: SqliteStore) -> None:
    """A pipeline mismatch is rebuilt in place. Query still refuses a stale index."""
    try:
        index.open()
    except IngestError as exc:
        if "pipeline version changed" not in str(exc):
            raise
        index.close()
        _wipe_index(Path(index.path))
        index.open()


def _wipe_index(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def ingest(
    path: str | Path,
    index_dir: str | Path,
    encode: Callable[..., list[list[float]]] | None = None,
    count_tokens: Callable[[str], int] | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    *,
    contextual: bool = False,
    context_fn: Callable[[str, str], str] | None = None,
) -> list[Ingested]:
    from rag.telemetry import bind_trace, log_event, mint_trace_id

    path = Path(path)
    index_dir = Path(index_dir)
    model_id = model_id or EMBED_MODEL
    model_revision = model_revision or EMBED_REVISION
    from rag.embed import count_tokens as default_tokens
    from rag.embed import encode_documents as default_encode

    encoder: Callable[..., list[list[float]]] = default_encode if encode is None else encode
    token_fn: Callable[[str], int] = default_tokens if count_tokens is None else count_tokens
    if not path.exists():
        raise IngestError(str(path), "file not found")
    if path.is_file() and path.suffix.lower() not in MIMES:
        raise IngestError(str(path), "unsupported file type")
    index = Index(index_dir, model_id, model_revision, PIPELINE_VERSION)
    _open_for_ingest(index)
    try:
        files = _files(path, index_dir)
        results: list[Ingested] = []
        with bind_trace(mint_trace_id()):
            log_event("ingest_start", file_total=len(files))
            for file_index, (file, root) in enumerate(files, start=1):
                log_event(
                    "ingest_progress",
                    file=str(file.name),
                    file_index=file_index,
                    file_total=len(files),
                )
                results.append(
                    _ingest_one(
                        file,
                        root,
                        index,
                        encoder,
                        token_fn,
                        model_id,
                        model_revision,
                        contextual=contextual,
                        context_fn=context_fn,
                        file_index=file_index,
                        file_total=len(files),
                    )
                )
            if path.is_dir():
                seen = {item.doc_id for item in results}
                for doc_id in index.purge_missing(list(seen)):
                    results.append(Ingested(doc_id, "purged", 0))
            from rag.versions import reconcile_versions

            reconcile_versions(index)
            log_event("ingest_done", files=len(results))
        return results
    finally:
        index.close()


def query(
    text: str,
    index_dir: str | Path,
    doc_id: str | None = None,
    embed_query: Callable[[str], list[float]] | None = None,
    rerank: Callable[[str, list[str]], list[float]] | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    mode: str = "cascade",
) -> Retrieval:
    from rag.models import QueryError

    if not text or not text.strip():
        raise QueryError("empty query")
    model_id = model_id or EMBED_MODEL
    model_revision = model_revision or EMBED_REVISION
    from rag.embed import encode_query, rerank_scores

    query_embed: Callable[[str], list[float]] = encode_query if embed_query is None else embed_query
    query_rerank: Callable[[str, list[str]], list[float]] = (
        rerank_scores if rerank is None else rerank
    )
    index = Index(Path(index_dir), model_id, model_revision, PIPELINE_VERSION)
    index.open()
    try:
        return retrieve(index, text, query_embed, rerank=query_rerank, doc_id=doc_id, mode=mode)
    finally:
        index.close()


def _ingest_one(
    file: Path,
    root: Path,
    index: SqliteStore,
    encode: Callable[..., list[list[float]]],
    count_tokens: Callable[[str], int],
    model_id: str,
    model_revision: str,
    *,
    contextual: bool = False,
    context_fn: Callable[[str, str], str] | None = None,
    file_index: int = 1,
    file_total: int = 1,
) -> Ingested:
    from rag.normalize import normalize_text
    from rag.telemetry import log_event, span

    resolved = _inside(root, file)
    if resolved.stat().st_size > MAX_FILE_BYTES:
        raise IngestError(str(file), "file exceeds 50 MB")
    digest = _sha256(resolved)
    doc_id = resolved.relative_to(root.resolve()).as_posix()
    if index.matches(doc_id, digest, PIPELINE_VERSION, model_id, model_revision):
        log_event(
            "ingest_skip",
            file=doc_id,
            file_index=file_index,
            file_total=file_total,
            cache_hit=True,
        )
        return Ingested(doc_id, "skipped", 0)
    with span("parse", file=doc_id, file_index=file_index, file_total=file_total) as parse_span:
        mime, blocks = parse_file(resolved)
        pages = {block.page for block in blocks if getattr(block, "page", None)}
        parse_span["page_total"] = len(pages) or 1
        layout_escalations = sum(1 for block in blocks if block.flagged and block.kind == "layout")
        ocr_pages = sum(1 for block in blocks if getattr(block, "ocr", False) or block.kind == "ocr")
        parse_span["layout_escalations"] = layout_escalations
        parse_span["ocr_pages"] = ocr_pages
        for page in sorted(pages) if pages else [1]:
            log_event(
                "ingest_page",
                file=doc_id,
                file_index=file_index,
                file_total=file_total,
                page=page,
                page_total=parse_span["page_total"],
            )
    with span("normalize", file=doc_id):
        from dataclasses import replace

        blocks = [
            replace(block, text=normalize_text(block.text, code=block.kind == "code"))
            for block in blocks
        ]
    warnings: list[str] = []
    for block in blocks:
        if block.flagged:
            warnings.append(f"flagged {block.kind} on page {block.page}")
        if block.text.startswith("[page error]"):
            warnings.append(block.text)
    ctx = context_fn if contextual else None
    with span("chunk", file=doc_id) as chunk_span:
        _parents, children = chunk_document(doc_id, blocks, count_tokens, context_fn=None)
        chunk_span["chunks"] = len(children)
    with span("contextualize", file=doc_id, skipped=not bool(contextual)) as ctx_span:
        if contextual:
            _parents, children = chunk_document(doc_id, blocks, count_tokens, context_fn=ctx)
            ctx_span["chunks"] = len(children)
        else:
            ctx_span["reason"] = "contextual_disabled"
    if not children:
        raise IngestError(str(file), "no chunks")
    if len(children) > MAX_CHUNKS:
        raise IngestError(str(file), "parser produced too many chunks")
    with span("embed", file=doc_id, model=model_id, candidates_in=len(children)) as embed_span:
        vectors = embed_texts(
            [child.embed_text for child in children],
            encode,
            cache_get=index.cache_get,
            cache_put=index.cache_put,
            model_id=model_id,
            revision=model_revision,
        )
        embed_span["candidates_out"] = len(vectors)
        # cache hit rate when embed_texts reports via cache — approximate from sizes
        embed_span["cache_hit_rate"] = None
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
    log_event(
        "ingest_file_done",
        file=doc_id,
        file_index=file_index,
        file_total=file_total,
        chunks=len(children),
        layout_escalations=layout_escalations,
        ocr_pages=ocr_pages,
    )
    return Ingested(doc_id, "indexed", len(children), warnings=warnings or None)


def _files(path: Path, index_dir: Path) -> list[tuple[Path, Path]]:
    if path.is_dir():
        root = path
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            current = Path(dirpath)
            kept_dirs = []
            for name in dirnames:
                child = current / name
                if child.is_symlink():
                    _check_symlink(root, child)
                    continue
                kept_dirs.append(name)
            dirnames[:] = kept_dirs
            for name in filenames:
                file = current / name
                _reject_name(file)
                if file.is_symlink():
                    _check_symlink(root, file)
                if file.name.startswith(".") or not file.is_file():
                    continue
                if file.suffix.lower() not in MIMES:
                    continue
                if _is_under(index_dir, file):
                    continue
                found.append(file)
        return [(file, root) for file in sorted(found)]
    _reject_name(path)
    if path.suffix.lower() not in MIMES:
        raise IngestError(str(path), "unsupported file type")
    return [(path, path.parent)]


def _reject_name(file: Path) -> None:
    name = file.name
    if "\n" in name or "\r" in name:
        raise IngestError(str(file), f"filename contains a newline: {file}")
    if ".." in file.parts or "../" in name:
        raise IngestError(str(file), f"path escapes the ingest folder: {file}")


def _check_symlink(root: Path, file: Path) -> None:
    try:
        resolved = file.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IngestError(str(file), f"symlink loop: {file}") from exc
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise IngestError(str(file), f"symlink escapes the ingest folder: {file}")
    if resolved == root_resolved or resolved in file.parents:
        raise IngestError(str(file), f"symlink loop: {file}")


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
