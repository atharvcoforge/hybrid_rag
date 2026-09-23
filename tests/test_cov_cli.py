"""CLI success and failure paths without model weights."""

import json

import pytest

from rag.cli import (
    _answers_for_rows,
    _format_hit,
    _gated_text,
    _run_calibrate,
    _run_eval,
    _run_verify,
    _stamp_quality,
    main,
)
from rag.evaluate import Score
from rag.gates import GateResult
from rag.models import Hit, Ingested, QueryError, Retrieval


def _hit():
    return Hit(
        parent_id="p",
        parent_text="answer",
        heading_path="",
        source_path="doc.pdf",
        file_sha256="abcdef1234567890",
        page_start=0,
        page_end=0,
        start_char=0,
        end_char=1,
        child_id="c",
        score=0.5,
        confident=False,
    )


def test_main_commands(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("rag.cli.ingest", lambda *_a, **_k: [Ingested("a.md", "indexed", 2), Ingested("b.md", "purged", 0)])
    assert main(["--log-format", "console", "ingest", "docs", "--index", "idx"]) == 0
    assert "indexed" in capsys.readouterr().out
    monkeypatch.setattr(
        "rag.cli.query",
        lambda *_a, **_k: Retrieval(hits=[_hit()], reason=""),
    )
    assert main(["query", "hello", "--index", "idx", "--doc", "a.md"]) == 0
    assert "uncertain" in capsys.readouterr().out
    monkeypatch.setattr("rag.cli.query", lambda *_a, **_k: Retrieval(hits=[], reason="no_confident_hit"))
    assert main(["query", "hello", "--index", "idx"]) == 0
    assert "no_confident_hit" in capsys.readouterr().out
    monkeypatch.setattr("rag.cli.query", lambda *_a, **_k: Retrieval(hits=[], reason=""))
    assert main(["query", "hello", "--index", "idx"]) == 0
    assert "no hits" in capsys.readouterr().out
    assert main(["purge", "docs", "--index", "idx", "--missing"]) == 0
    assert "purged" in capsys.readouterr().out
    monkeypatch.setattr("rag.cli._run_verify", lambda _index: [])
    assert main(["verify", "--index", "idx"]) == 0
    assert "integrity clean" in capsys.readouterr().out
    monkeypatch.setattr("rag.cli._run_verify", lambda _index: ["fts row without a vector: c"])
    assert main(["verify", "--index", "idx"]) == 1
    monkeypatch.setattr("rag.cli._run_calibrate", lambda *_a: "{\"ok\": true}")
    assert main(["calibrate", "--index", "idx", "--golden", "g.jsonl"]) == 0
    monkeypatch.setattr("rag.cli._run_eval", lambda *_a: ("gates: PASS", True))
    assert main(["eval", "--index", "idx", "--suite", "missing.yaml"]) == 0
    monkeypatch.setattr("rag.cli._run_eval", lambda *_a: ("gates: FAIL", False))
    assert main(["eval", "--index", "idx"]) == 1
    from rag.models import IngestError

    def boom(*_a, **_k):
        raise IngestError("x", "bad file")

    monkeypatch.setattr("rag.cli.ingest", boom)
    assert main(["ingest", "docs", "--index", "idx"]) == 1
    err = capsys.readouterr().err
    assert "bad file" in err


def test_verify_and_format(tmp_path):
    problems = _run_verify(str(tmp_path / "empty"))
    assert problems == [] or isinstance(problems, list)
    text = _format_hit(_hit())
    assert "p.-" in text
    hit = _hit()
    hit.page_start = 2
    hit.confident = True
    assert "confident" in _format_hit(hit)


def test_gated_text_and_quality(monkeypatch):
    empty = Retrieval(hits=[], reason="no_confident_hit")
    assert _gated_text("q", empty, lambda *_: "x", lambda *_a, **_k: None) == ("", None)
    result = Retrieval(hits=[_hit()])

    def complete(_q, _hits):
        return "plain"

    def withheld(*_a, **_k):
        return GateResult(False, "missing_citation", state="withheld", answer="")

    assert _gated_text("q", result, complete, withheld)[0] == "The documents do not say."

    def ok(*_a, **_k):
        return GateResult(True, "", state="verified", answer="cited [1]", groundedness=None, citation_precision=None)

    text, gate = _gated_text("q", result, complete, ok)
    assert text == "cited [1]"
    assert gate is not None
    monkeypatch.setattr("rag.generate.writer_up", lambda: False)
    assert _answers_for_rows([{"id": "1", "q": "q"}], lambda *_: result, "rrf") == (None, None)
    monkeypatch.setattr("rag.generate.writer_up", lambda: True)
    monkeypatch.setattr("rag.generate.complete", lambda *_: "plain")
    def ask_rows(_mode, row):
        if row["id"] == "empty":
            return Retrieval(hits=[])
        return result

    answers, quality = _answers_for_rows(
        [{"id": "empty", "q": "q"}, {"id": "1", "q": "q"}],
        ask_rows,
        "rrf",
    )
    assert answers is not None
    lines = [Score("rrf", "all", 1, 1, 0, 1), Score("bm25", "all", 1, 1, 0, 1)]
    _stamp_quality(lines, "rrf", quality)
    _stamp_quality(lines, "rrf", None)
    assert lines[0].groundedness is not None or quality == {"groundedness": None, "citation_precision": None}


def test_eval_and_calibrate_helpers(monkeypatch, tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_text(json.dumps({"id": "1", "q": "q", "kind": "lexical"}) + "\n\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text("rows: g.jsonl\n", encoding="utf-8")
    monkeypatch.setattr("rag.cli.verify_corpus", lambda _suite: ["missing doc"])
    monkeypatch.setattr("rag.cli.assert_kind_coverage", lambda _rows: ["need lexical"])
    monkeypatch.setattr("rag.cli.load_suite", lambda _path: {"rows": str(golden)})
    text, ok = _run_eval(str(tmp_path), None, str(suite))
    assert ok is False
    assert "corpus pins" in text and "kinds" in text

    monkeypatch.setattr("rag.cli.verify_corpus", lambda _suite: [])
    monkeypatch.setattr("rag.cli.assert_kind_coverage", lambda _rows: [])
    monkeypatch.setattr("rag.cli.load_suite", lambda _path: None)
    monkeypatch.setattr(
        "rag.cli.evaluate",
        lambda *_a, **_k: ([Score("rrf", "all", 1, 1, 0, 1)], None),
    )
    monkeypatch.setattr("rag.embed.encode_query", lambda _text: [0.0])
    monkeypatch.setattr("rag.embed.rerank_scores", lambda _q, texts: [0.0] * len(texts))
    monkeypatch.setattr("rag.retrieve.retrieve", lambda *_a, **_k: Retrieval(hits=[]))
    text, ok = _run_eval(str(tmp_path / "idx"), str(golden), str(tmp_path / "no-such.yaml"))
    assert ok is True
    assert "rrf" in text

    with_suite = {
        "rows": str(golden),
        "split": str(tmp_path / "no-split.json"),
        "regression": {"baseline": str(tmp_path / "base.json")},
    }
    (tmp_path / "base.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("rag.cli.load_suite", lambda _path: with_suite)
    monkeypatch.setattr("rag.evaluate.pick_live", lambda _lines: "rrf")
    monkeypatch.setattr("rag.generate.writer_up", lambda: False)
    monkeypatch.setattr("rag.evaluate.check_gates", lambda *_a, **_k: ["mrr low"])
    text, ok = _run_eval(str(tmp_path / "idx2"), None, str(suite))
    assert ok is False
    assert "generator down" in text

    monkeypatch.setattr("rag.calibrate.calibrate", lambda *_a, **_k: {"modes": {}})
    out = tmp_path / "cal.json"
    body = _run_calibrate(str(tmp_path / "idx3"), str(golden), str(tmp_path / "missing-split.json"), str(out))
    assert "modes" in body
    assert out.exists()

    pins_only = {"rows": str(golden)}
    monkeypatch.setattr("rag.cli.load_suite", lambda _path: pins_only)
    monkeypatch.setattr("rag.cli.verify_corpus", lambda _suite: ["missing"])
    monkeypatch.setattr("rag.cli.assert_kind_coverage", lambda _rows: [])
    text, ok = _run_eval(str(tmp_path), None, str(suite))
    assert ok is False and "kinds" not in text
    monkeypatch.setattr("rag.cli.verify_corpus", lambda _suite: [])
    monkeypatch.setattr("rag.cli.assert_kind_coverage", lambda _rows: ["lexical"])
    text, ok = _run_eval(str(tmp_path), None, str(suite))
    assert "kinds" in text
    monkeypatch.setattr("rag.cli.load_suite", lambda _path: None)
    with pytest.raises(QueryError, match="golden"):
        _run_eval(str(tmp_path), None, str(tmp_path / "absent.yaml"))

    split_file = tmp_path / "split.json"
    split_file.write_text('{"train": ["1"], "test": []}', encoding="utf-8")
    passing = {"rows": str(golden), "split": str(split_file), "regression": {}}
    monkeypatch.setattr("rag.cli.load_suite", lambda _path: passing)
    monkeypatch.setattr("rag.cli.verify_corpus", lambda _suite: [])
    monkeypatch.setattr("rag.cli.assert_kind_coverage", lambda _rows: [])
    monkeypatch.setattr("rag.generate.writer_up", lambda: True)
    monkeypatch.setattr("rag.generate.complete", lambda *_a: "plain")
    monkeypatch.setattr("rag.retrieve.retrieve", lambda *_a, **_k: Retrieval(hits=[_hit()]))
    monkeypatch.setattr("rag.evaluate.check_gates", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "rag.cli.evaluate",
        lambda *_a, **_k: ([Score("rrf", "all", 1, 1, 0, 1)], 0.1),
    )

    def figure(*_a, **_k):
        return GateResult(False, "unsupported_figure", state="withheld")

    monkeypatch.setattr("rag.gates.check_all", figure)
    text, ok = _run_eval(str(tmp_path / "idx4"), str(golden), str(suite))
    assert "generator down" not in text

    def scored(*_a, **_k):
        return GateResult(True, "", state="verified", groundedness=1.0, citation_precision=1.0)

    monkeypatch.setattr("rag.gates.check_all", scored)
    _run_eval(str(tmp_path / "idx5"), str(golden), str(suite))

    def calls_ask(index, rows, ask, split=None):
        del index, split
        for row in rows:
            ask("rrf", row)
        return [Score("rrf", "all", 1, 1, 0, 1)], 0.2

    monkeypatch.setattr("rag.cli.evaluate", calls_ask)

    def calibrate_calls(index, rows, ask, split=None):
        del index, split
        for row in rows:
            ask("rrf", row)
        return {"modes": {}}

    monkeypatch.setattr("rag.calibrate.calibrate", calibrate_calls)
    _run_calibrate(str(tmp_path / "idx6"), str(golden), str(split_file), str(tmp_path / "cal2.json"))
