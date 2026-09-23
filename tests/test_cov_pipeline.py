import os
from collections.abc import Callable
from pathlib import Path

import pytest

from rag.models import PIPELINE_VERSION, Block, Ingested, IngestError, QueryError
from rag.pipeline import (
    _files,
    _inside,
    _is_under,
    _open_for_ingest,
    _reject_name,
    _wipe_index,
    ingest,
    query,
)
from rag.store import SqliteStore
from tests.fakes import fake_encode, fake_tokens


def _ingest(
    path: Path,
    index_dir: Path,
    *,
    encode: Callable[..., list[list[float]]] = fake_encode,
    count_tokens: Callable[[str], int] = fake_tokens,
    model_id: str = "test-embed",
    model_revision: str = "rev",
    contextual: bool = False,
    context_fn: Callable[[str, str], str] | None = None,
) -> list[Ingested]:
    return ingest(
        path,
        index_dir,
        encode=encode,
        count_tokens=count_tokens,
        model_id=model_id,
        model_revision=model_revision,
        contextual=contextual,
        context_fn=context_fn,
    )


def _encode_query(text: str) -> list[float]:
    return fake_encode([text], query=True)[0]


def _rerank(query_text: str, texts: list[str]) -> list[float]:
    del query_text
    return [1.0 for _ in texts]


def _note(folder: Path, name: str = "note.md", body: str = "alpha phrase") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(f"# Title\n\n{body}\n", encoding="utf-8")
    return path


def test_wipe_missing_path_and_name_rejects(tmp_path: Path) -> None:
    missing = tmp_path / "absent"
    _wipe_index(missing)
    assert not missing.exists()

    with pytest.raises(IngestError, match="newline"):
        _reject_name(Path("bad\nname.md"))
    with pytest.raises(IngestError, match="newline"):
        _reject_name(Path("bad\rname.md"))
    with pytest.raises(IngestError, match="escapes"):
        _reject_name(Path("documents/../secret.md"))

    outside = tmp_path / "outside.md"
    outside.write_text("x", encoding="utf-8")
    root = tmp_path / "docs"
    root.mkdir()
    inside = root / "inside.md"
    inside.write_text("y", encoding="utf-8")
    with pytest.raises(IngestError, match="unreadable"):
        _inside(root, root / "missing.md")
    with pytest.raises(IngestError, match="escapes"):
        _inside(root, outside)
    assert _inside(root, inside) == inside.resolve()
    assert _inside(root, root) == root.resolve()
    assert _is_under(root, inside) is True
    assert _is_under(root / "index", inside) is False

    binary = tmp_path / "blob.bin"
    binary.write_text("nope", encoding="utf-8")
    with pytest.raises(IngestError, match="unsupported"):
        _files(binary, tmp_path / "index")


def test_ingest_gates_and_single_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"
    with pytest.raises(IngestError, match="file not found"):
        _ingest(missing, tmp_path / "index-missing")

    binary = tmp_path / "blob.bin"
    binary.write_text("nope", encoding="utf-8")
    with pytest.raises(IngestError, match="unsupported"):
        _ingest(binary, tmp_path / "index-bin")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert _ingest(empty_dir, tmp_path / "index-empty") == []

    note = _note(tmp_path, "note.md")
    index = tmp_path / "index-file"
    first = _ingest(note, index)
    assert first[0].status == "indexed"
    assert first[0].warnings is None
    second = _ingest(note, index)
    assert second[0].status == "skipped"

    blank = tmp_path / "blank.md"
    blank.write_text("", encoding="utf-8")
    with pytest.raises(IngestError, match="no text"):
        _ingest(blank, tmp_path / "index-blank")


def test_ingest_refuses_a_document_that_chunks_to_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag.pipeline.chunk_document", lambda *_args, **_kwargs: ([], []))
    note = _note(tmp_path, "empty-chunks.md")
    with pytest.raises(IngestError, match="no chunks"):
        _ingest(note, tmp_path / "index-none")


def test_size_and_chunk_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    big = _note(tmp_path, "big.md", "too big")
    monkeypatch.setattr("rag.pipeline.MAX_FILE_BYTES", 4)
    with pytest.raises(IngestError, match="50 MB"):
        _ingest(big, tmp_path / "index-big")

    monkeypatch.setattr("rag.pipeline.MAX_FILE_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr("rag.pipeline.MAX_CHUNKS", 0)
    small = _note(tmp_path, "small.md")
    with pytest.raises(IngestError, match="too many chunks"):
        _ingest(small, tmp_path / "index-chunks")


def test_directory_filters_symlink_and_purge(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "keep.md").write_text("# Keep\n\nkeep this phrase\n", encoding="utf-8")
    inner = root / "inner"
    inner.mkdir()
    (inner / "also.md").write_text("# Also\n\nalso this phrase\n", encoding="utf-8")
    (root / ".hidden.md").write_text("# Hide\n\nhidden phrase\n", encoding="utf-8")
    (root / "notes.log").write_text("log", encoding="utf-8")
    os.mkfifo(root / "pipe.md")
    (root / "linked").symlink_to(inner, target_is_directory=True)
    (root / "real.md").write_text("# Real\n\nreal phrase\n", encoding="utf-8")
    (root / "alias.md").symlink_to(root / "real.md")
    index = root / "index"
    index.mkdir()
    (index / "nested.md").write_text("# Nested\n\nnested phrase\n", encoding="utf-8")

    items = _ingest(root, index)
    ids = {item.doc_id for item in items}
    assert "keep.md" in ids
    assert "inner/also.md" in ids
    assert "real.md" in ids
    assert "alias.md" not in ids
    assert ".hidden.md" not in ids
    assert "notes.log" not in ids
    assert "pipe.md" not in ids
    assert "index/nested.md" not in ids
    assert "linked/also.md" not in ids

    (root / "keep.md").unlink()
    again = _ingest(root, index)
    assert any(item.status == "purged" and item.doc_id == "keep.md" for item in again)


def test_symlink_escape_and_loops(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "note.md").write_text("# T\n\nhello\n", encoding="utf-8")
    outside = tmp_path / "secret.md"
    outside.write_text("# S\n\nsecret\n", encoding="utf-8")
    (root / "escape.md").symlink_to(outside)
    with pytest.raises(IngestError, match="escapes"):
        _ingest(root, tmp_path / "index-escape")

    loop_root = tmp_path / "loop"
    loop_root.mkdir()
    (loop_root / "note.md").write_text("# T\n\nhello\n", encoding="utf-8")
    (loop_root / "up").symlink_to(loop_root, target_is_directory=True)
    with pytest.raises(IngestError, match="loop"):
        _ingest(loop_root, tmp_path / "index-up")

    parent_loop = tmp_path / "parent"
    parent_loop.mkdir()
    sub = parent_loop / "sub"
    sub.mkdir()
    (sub / "note.md").write_text("# T\n\nhello\n", encoding="utf-8")
    (sub / "up").symlink_to(sub, target_is_directory=True)
    with pytest.raises(IngestError, match="loop"):
        _ingest(parent_loop, tmp_path / "index-parent")

    self_loop = tmp_path / "self"
    self_loop.mkdir()
    (self_loop / "note.md").write_text("# T\n\nhello\n", encoding="utf-8")
    broken = self_loop / "loop.md"
    broken.symlink_to(broken)
    with pytest.raises(IngestError, match="loop"):
        _ingest(self_loop, tmp_path / "index-self")


def test_flagged_pages_and_contextual(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    note = _note(tmp_path, "flag.md")

    def parse_file(path: Path) -> tuple[str, list[Block]]:
        del path
        return "text/markdown", [
            Block("layout", "broken table row", "H", 2, 0, 16, flagged=True),
            Block("prose", "flagged prose", "H", 2, 0, 13, flagged=True),
            Block("ocr", "scanned line", "H", 2, 0, 12),
            Block("prose", "optical line", "H", 2, 0, 12, ocr=True),
            Block("prose", "[page error] page 2: boom", "", 2, 0, 24),
            Block("prose", "visible body text", "H", 0, 0, 18),
        ]

    monkeypatch.setattr("rag.pipeline.parse_file", parse_file)
    items = _ingest(
        note,
        tmp_path / "index-flag",
        contextual=True,
        context_fn=lambda _parent, _child: "context",
    )
    warnings = items[0].warnings or []
    assert any(item.startswith("flagged layout") for item in warnings)
    assert any(item.startswith("flagged prose") for item in warnings)
    assert any(item.startswith("[page error]") for item in warnings)
    assert items[0].status == "indexed"


def test_identity_wipe_rules(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    note = _note(folder)
    index = tmp_path / "index"
    _ingest(note.parent, index)
    marker = index / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(IngestError, match="delete the index directory to rebuild") as exc:
        _ingest(note.parent, index, model_id="other-embed")
    assert "pipeline version changed" not in str(exc.value)
    assert marker.exists()

    with SqliteStore(index, "test-embed", "rev", PIPELINE_VERSION) as store:
        store._set_meta("pipeline_version", "1")
    versioned = SqliteStore(index, "test-embed", "rev", PIPELINE_VERSION)
    with pytest.raises(
        IngestError,
        match="pipeline version changed from 1 to 3; delete the index directory to rebuild",
    ):
        versioned.open()
    versioned.close()

    (index / "extra").mkdir()
    (index / "extra" / "a.txt").write_text("a", encoding="utf-8")
    (index / "loose.txt").write_text("b", encoding="utf-8")
    (index / "linkdir").symlink_to(index / "extra", target_is_directory=True)
    rebuilt = _ingest(note.parent, index)
    assert rebuilt[0].status == "indexed"
    assert not (index / "extra").exists()
    assert not (index / "loose.txt").exists()
    assert not (index / "linkdir").exists()
    assert not marker.exists()


def test_open_for_ingest_refuses_model_mismatch(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    _ingest(_note(folder).parent, tmp_path / "index")
    index = SqliteStore(tmp_path / "index", "other-embed", "rev", PIPELINE_VERSION)
    with pytest.raises(IngestError, match="delete the index directory to rebuild"):
        _open_for_ingest(index)
    assert (tmp_path / "index" / "rag.sqlite").exists()


def test_query_and_default_callables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "docs"
    index = tmp_path / "index"
    _ingest(_note(folder).parent, index)
    with pytest.raises(QueryError):
        query(
            "",
            index,
            embed_query=_encode_query,
            rerank=_rerank,
            model_id="test-embed",
            model_revision="rev",
        )
    with pytest.raises(QueryError):
        query(
            "   ",
            index,
            embed_query=_encode_query,
            rerank=_rerank,
            model_id="test-embed",
            model_revision="rev",
        )
    found = query(
        "alpha",
        index,
        embed_query=_encode_query,
        rerank=_rerank,
        model_id="test-embed",
        model_revision="rev",
    )
    assert found.hits is not None

    monkeypatch.setattr("rag.embed.encode_documents", fake_encode)
    monkeypatch.setattr("rag.embed.count_tokens", fake_tokens)
    monkeypatch.setattr("rag.embed.encode_query", _encode_query)
    monkeypatch.setattr("rag.embed.rerank_scores", _rerank)
    other = tmp_path / "plain"
    plain_index = tmp_path / "index-plain"
    items = ingest(_note(other).parent, plain_index)
    assert items[0].status == "indexed"
    assert query("alpha", plain_index).hits is not None
