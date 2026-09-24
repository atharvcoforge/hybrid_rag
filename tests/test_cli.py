from rag.cli import _live_mode, _run_ask, main
from rag.models import EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION, Hit, Retrieval


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


def test_ask_cites_the_passage_and_falls_back_to_rrf(tmp_path, monkeypatch):
    from rag.store import Index

    index = Index(tmp_path / "idx", EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION)
    index.open()
    index.close()
    assert _live_mode(str(tmp_path / "idx")) == "rrf"

    hit = Hit(
        "p",
        "Publication date: 10 October 2025",
        "Plan",
        "Carbon_Reduction_Plan.pdf",
        "abc",
        1,
        1,
        0,
        20,
        "c",
        0.9,
        True,
    )
    monkeypatch.setattr("rag.cli.query", lambda *_a, **_k: Retrieval(hits=[hit]))
    monkeypatch.setattr(
        "rag.generate.complete",
        lambda *_a, **_k: "The plan was published on 10 October 2025 [1].",
    )
    assert main(["ask", "When was it published?", "--index", str(tmp_path / "idx"), "--mode", "rrf"]) == 0
    text = _run_ask("When was it published?", str(tmp_path / "idx"), "rrf")
    assert "10 October 2025" in text
    assert "[1] CURRENT" in text
    stale = Hit(
        "p2",
        "old",
        "H",
        "old.pdf",
        "abc",
        1,
        2,
        0,
        3,
        "c2",
        0.1,
        False,
        superseded=True,
        status="superseded",
    )
    monkeypatch.setattr("rag.cli.query", lambda *_a, **_k: Retrieval(hits=[stale]))
    monkeypatch.setattr("rag.generate.complete", lambda *_a, **_k: "The documents do not say.")
    stale_text = _run_ask("q", str(tmp_path / "idx"), "dense")
    assert "SUPERSEDED" in stale_text
    monkeypatch.setattr("rag.cli.query", lambda *_a, **_k: Retrieval(hits=[], reason="no_confident_hit"))
    assert _run_ask("q", str(tmp_path / "idx"), "dense") == "no_confident_hit"
    monkeypatch.setattr("rag.cli.query", lambda *_a, **_k: Retrieval(hits=[]))
    assert _run_ask("q", str(tmp_path / "idx"), "bm25") == "no hits"


def test_unsupported_file_is_refused(tmp_path, capsys):
    bad = tmp_path / "notes.exe"
    bad.write_text("nope", encoding="utf-8")
    assert main(["ingest", str(bad), "--index", str(tmp_path / "index")]) == 1
    assert "unsupported" in capsys.readouterr().err
