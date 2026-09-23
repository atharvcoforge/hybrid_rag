from rag.cli import main


def test_empty_query_is_refused(capsys):
    assert main(["query", "   ", "--index", "unused"]) == 1
    assert "empty query" in capsys.readouterr().err


def test_verify_reports_a_parent_with_no_document(tmp_path, capsys):
    index = tmp_path / "index"
    assert main(["verify", "--index", str(index)]) == 0
    assert "integrity clean" in capsys.readouterr().out

    from rag.models import EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION
    from rag.store import Index

    store = Index(index, EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION)
    store.open()
    store.db.execute(
        """
        INSERT INTO parents (
            parent_id, doc_id, text, heading_path, block_type,
            page_start, page_end, norm_start, norm_end, parent_index, token_count
        ) VALUES ('orphan', 'missing.pdf', 'text', '', 'prose', 0, 0, 0, 0, 0, 1)
        """
    )
    store.db.commit()
    store.close()
    assert main(["verify", "--index", str(index)]) == 1
    assert "parent without a document: orphan" in capsys.readouterr().err


def test_unsupported_file_is_refused(tmp_path, capsys):
    bad = tmp_path / "notes.exe"
    bad.write_text("nope", encoding="utf-8")
    assert main(["ingest", str(bad), "--index", str(tmp_path / "index")]) == 1
    assert "unsupported" in capsys.readouterr().err
