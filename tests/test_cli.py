from rag.cli import main


def test_empty_query_is_refused(capsys):
    assert main(["query", "   ", "--index", "unused"]) == 1
    assert "empty query" in capsys.readouterr().err


def test_unsupported_file_is_refused(tmp_path, capsys):
    bad = tmp_path / "notes.exe"
    bad.write_text("nope", encoding="utf-8")
    assert main(["ingest", str(bad), "--index", str(tmp_path / "index")]) == 1
    assert "unsupported" in capsys.readouterr().err
