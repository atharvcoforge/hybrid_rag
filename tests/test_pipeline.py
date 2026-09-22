import re
from pathlib import Path

import pytest

from rag.evaluate import evaluate, load_rows
from rag.models import IngestError, PIPELINE_VERSION
from rag.pipeline import ingest, query
from rag.retrieve import retrieve
from rag.store import Index
from tests.fakes import fake_encode, fake_tokens

CORPUS = Path("evals/corpus")


def _encode_query(text):
    return fake_encode([text], query=True)[0]


def _keyword(query_text, texts):
    wanted = set(re.findall(r"\w+", query_text.lower()))
    scores = []
    for text in texts:
        words = set(re.findall(r"\w+", text.lower()))
        scores.append(float(len(wanted & words)))
    return scores


def _ingest(path, index_dir, encode=None):
    return ingest(
        path,
        index_dir,
        encode=encode or fake_encode,
        count_tokens=fake_tokens,
        model_id="test-embed",
        model_revision="rev",
    )


def test_sku_is_found_and_a_second_ingest_skips(tmp_path):
    index_dir = tmp_path / "index"
    calls = []

    def encode(texts, *, query=False):
        calls.append(list(texts))
        return fake_encode(texts, query=query)

    first = _ingest(CORPUS, index_dir, encode)
    assert any(item.status == "indexed" and item.doc_id == "guide.md" for item in first)
    second = _ingest(CORPUS, index_dir, encode)
    assert second[0].status == "skipped"
    assert len(calls) == 1

    result = query(
        "SKU-7842-XL",
        index_dir,
        embed_query=_encode_query,
        rerank=_keyword,
        model_id="test-embed",
        model_revision="rev",
    )
    assert result.hits
    assert "SKU-7842-XL" in result.hits[0].parent_text
    assert result.hits[0].file_sha256
    index = Index(index_dir, "test-embed", "rev", PIPELINE_VERSION)
    index.open()
    try:
        assert index.bm25_search("SKU-7842-XL", 5, doc_id="guide.md")
        assert index.dense_search(_encode_query("SKU-7842-XL"), 5, doc_id="missing.md") == []
    finally:
        index.close()


def test_replacing_a_file_drops_the_old_chunks(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    path = folder / "note.md"
    path.write_text("# One\n\nalpha unique phrase\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    _ingest(folder, index_dir)
    index = Index(index_dir, "test-embed", "rev", PIPELINE_VERSION)
    index.open()
    try:
        before = set(index.child_ids("note.md"))
    finally:
        index.close()
    path.write_text("# One\n\nbeta unique phrase\n", encoding="utf-8")
    _ingest(folder, index_dir)
    index.open()
    try:
        after = set(index.child_ids("note.md"))
    finally:
        index.close()
    assert before
    assert after
    assert before.isdisjoint(after)
    result = query(
        "alpha unique phrase",
        index_dir,
        embed_query=_encode_query,
        rerank=_keyword,
        model_id="test-embed",
        model_revision="rev",
    )
    assert not any("alpha unique phrase" in hit.parent_text for hit in result.hits)


def test_removing_a_file_purges_it_from_the_index(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    keep = folder / "keep.md"
    drop = folder / "drop.md"
    keep.write_text("# Keep\n\nkeep phrase unique\n", encoding="utf-8")
    drop.write_text("# Drop\n\ndrop phrase unique\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    _ingest(folder, index_dir)
    drop.unlink()
    items = _ingest(folder, index_dir)
    assert any(item.status == "purged" and item.doc_id == "drop.md" for item in items)
    result = query(
        "drop phrase unique",
        index_dir,
        embed_query=_encode_query,
        rerank=_keyword,
        model_id="test-embed",
        model_revision="rev",
    )
    assert not any("drop phrase unique" in hit.parent_text for hit in result.hits)


def test_a_different_model_refuses_the_index(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "note.md").write_text("# One\n\nhello\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    _ingest(folder, index_dir)
    with pytest.raises(IngestError, match="delete the index directory"):
        ingest(
            folder,
            index_dir,
            encode=fake_encode,
            count_tokens=fake_tokens,
            model_id="other-embed",
            model_revision="rev",
        )


def test_gates(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.pipeline.MAX_FILE_BYTES", 8)
    big = tmp_path / "big.md"
    big.write_text("this is too long", encoding="utf-8")
    with pytest.raises(IngestError, match="50 MB"):
        _ingest(big, tmp_path / "index-big")

    monkeypatch.setattr("rag.pipeline.MAX_FILE_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr("rag.pipeline.MAX_CHUNKS", 0)
    small = tmp_path / "small.md"
    small.write_text("# One\n\nhello\n", encoding="utf-8")
    with pytest.raises(IngestError, match="too many chunks"):
        _ingest(small, tmp_path / "index-chunks")

    outside = tmp_path / "outside.md"
    outside.write_text("# Secret\n\noutside\n", encoding="utf-8")
    folder = tmp_path / "docs"
    folder.mkdir()
    link = folder / "link.md"
    link.symlink_to(outside)
    with pytest.raises(IngestError, match="escapes"):
        _ingest(folder, tmp_path / "index-link")


def test_golden_bm25_finds_the_sku(tmp_path):
    index_dir = tmp_path / "index"
    _ingest(CORPUS, index_dir)
    rows = load_rows("evals/fixture.jsonl")
    index = Index(index_dir, "test-embed", "rev", PIPELINE_VERSION)
    index.open()
    try:
        def ask(mode, row):
            return retrieve(index, row["q"], _encode_query, rerank=_keyword, mode=mode)

        lines, fitted = evaluate(index, rows, ask)
    finally:
        index.close()
    lexical = next(line for line in lines if line.mode == "bm25" and line.kind == "lexical")
    assert lexical.recall == 1
    assert fitted is not None
