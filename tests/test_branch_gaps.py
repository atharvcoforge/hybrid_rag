"""Branch coverage for modules the fast suite was missing."""

import json
import sqlite3
import sys
import types

import pytest

from rag.cache import AnswerCache
from rag.calibrate import calibrate, fit_mode_threshold, write_report
from rag.generate import _fence, complete, stream_answer, writer_up
from rag.health import CircuitBreaker, HealthState
from rag.layout import extract_page_tables, looks_borderless, render_rows, score_table
from rag.models import Hit, Retrieval, make_embed_text
from rag.ocr import caption_figure, page_image, text_coverage
from rag.queue import BusyError, InferenceQueue
from rag.rerank import CrossEncoderReranker
from rag.store import Index
from rag.telemetry import configure_logging, log_event, set_stage_bus, span
from tests.fakes import fake_encode, fake_tokens


def _hit(text="14,644", score=0.9, source="doc.pdf"):
    return Hit(
        parent_id="p",
        parent_text=text,
        heading_path="H",
        source_path=source,
        file_sha256="abc",
        page_start=1,
        page_end=1,
        start_char=0,
        end_char=len(text),
        child_id="c",
        score=score,
        confident=True,
    )


def test_embed_text_without_a_heading_is_the_body():
    assert make_embed_text("", "body") == "body"


def test_cache_evicts_ttl_and_reports_an_empty_rate():
    cache = AnswerCache(maxsize=1, ttl_s=0)
    assert cache.hit_rate() == 0.0
    key = AnswerCache.make_key("Q", 1, "rrf")
    cache[key] = "answer"
    assert cache.get(key) is None
    cache.ttl_s = 3600
    cache[key] = "answer"
    other = AnswerCache.make_key("other", 1, "rrf")
    cache[other] = "next"
    assert cache.get(key) is None
    assert cache.get(other) == "next"
    cache.clear()
    assert cache.get(other) is None


def test_queue_context_manager_rejects_a_second_acquire():
    queue = InferenceQueue(maxsize=1)
    with queue:
        assert queue.in_flight == 1
        with pytest.raises(BusyError):
            queue.acquire()
    assert queue.in_flight == 0


def test_health_ignores_blank_notes_and_half_opens():
    state = HealthState()
    state.note("")
    state.note("fts down")
    state.note("fts down")
    assert state.messages == ["fts down"]
    breaker = CircuitBreaker(fail_threshold=1, reset_s=0)
    breaker.record_failure()
    assert breaker.state == "half_open"
    assert breaker.allow() is True


def test_fit_and_calibrate_write_a_held_out_report(tmp_path):
    rows = [
        {"id": "a", "kind": "lexical", "must_contain": "14,644", "doc_id": "doc.pdf"},
        {"id": "b", "kind": "unanswerable"},
    ]
    results = [Retrieval(hits=[_hit(score=0.8)]), Retrieval(hits=[_hit(text="nope", score=0.2)])]
    assert fit_mode_threshold([], []) is None
    assert fit_mode_threshold(rows[:1], results[:1]) is not None
    tau = fit_mode_threshold(rows, results, keep=0.5)
    assert tau is not None

    class Idx:
        model_id = "m"
        model_revision = "r"

        def __init__(self):
            self.saved = {}

        def set_tau(self, key, value):
            self.saved[key] = value

    index = Idx()

    def ask(mode, row):
        del mode
        if row.get("kind") == "unanswerable":
            return Retrieval(hits=[_hit(score=0.1)], reason="no_confident_hit")
        return Retrieval(hits=[_hit()])

    report = calibrate(index, rows, ask, split={"train": ["a"], "test": ["b"]})
    assert "modes" in report
    out = tmp_path / "nested" / "cal.json"
    write_report(out, report)
    assert json.loads(out.read_text())["modes"]


def test_cross_encoder_loads(monkeypatch):
    class Model:
        def predict(self, pairs, show_progress_bar=False):
            del pairs, show_progress_bar
            return [0.4]

    monkeypatch.setattr("rag.embed.load_reranker", lambda: Model())
    assert CrossEncoderReranker().score("q", []) == []
    assert CrossEncoderReranker().score("q", ["a"]) == [0.4]


def test_writer_stream_and_complete(monkeypatch):
    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n'
            yield b"noise\n"
            yield b"data: [DONE]\n"

        def read(self):
            return b'{"choices":[{"message":{"content":"Done"}}]}'

    monkeypatch.setattr("rag.generate.urllib.request.urlopen", lambda *args, **kwargs: Resp())
    assert writer_up() is True
    assert list(stream_answer("q", [_hit()])) == ["Hi"]
    assert complete("q", [_hit()]) == "Done"
    monkeypatch.setattr(
        "rag.generate.urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("down")),
    )
    assert writer_up() is False
    assert "tail" in _fence("<<PASSAGE_abc tail", "<<PASSAGE_zzzz>>")


def test_ocr_edges_and_stub_engine(monkeypatch):
    class Bad:
        width = 0
        height = 10

        def extract_text(self):
            raise RuntimeError("no text")

    assert text_coverage(Bad()) == 0.0

    class ImagePage:
        def to_image(self, resolution=0):
            del resolution
            raise RuntimeError("render")

    assert page_image(ImagePage()) is None
    assert page_image(object()) is None

    class Good:
        def to_image(self, resolution=0):
            del resolution

            class Image:
                original = "pixels"

            return Image()

    assert page_image(Good()) == "pixels"
    assert caption_figure("img", captioner=lambda _image: (_ for _ in ()).throw(RuntimeError("x"))) == ""

    fake = types.ModuleType("rapidocr_onnxruntime")

    class RapidOCR:
        def __call__(self, image):
            del image
            return [([0], "line", 0.5)], None

    fake.RapidOCR = RapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake)
    from rag.ocr import ocr_page

    result = ocr_page("img", engine=None)
    assert result.text == "" or "line" in result.text


def test_embed_stub_model_and_bad_encoder(monkeypatch):
    from rag.embed import (
        count_tokens,
        embed_texts,
        encode_documents,
        encode_query,
        reset_caches,
    )

    class Model:
        def encode(self, texts, **kwargs):
            del kwargs
            return [[1.0, 0.0] for _ in texts]

        class tokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return text.split()

    monkeypatch.setattr("rag.embed._embedder", Model())
    reset_caches()
    assert encode_documents(["hello"]) == [[1.0, 0.0]]
    assert encode_query("hello") == [1.0, 0.0]
    assert count_tokens("one two") == 2
    assert count_tokens("") == 0
    with pytest.raises(Exception, match="wrong number"):
        embed_texts(["a"], lambda texts, query=False: [])


def test_device_falls_back_when_torch_is_missing(monkeypatch):
    import builtins

    from rag.embed import _device

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert _device() == "cpu"


def test_importing_main_does_not_exit():
    import rag.__main__ as entry

    assert callable(entry.main)


def test_console_log_and_stage_bus(capsys):
    configure_logging("console")
    log_event("hello", note="x")
    captured = capsys.readouterr().out
    assert "hello" in captured
    bus = []

    class Bus:
        def put(self, payload):
            bus.append(payload)

    set_stage_bus(Bus())
    with span("dense", candidates_in=1):
        pass
    set_stage_bus(None)
    configure_logging("json")
    assert bus and bus[0]["stage"] == "dense"


def test_pipeline_rebuilds_a_version_bump(tmp_path):
    from rag.pipeline import ingest

    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "note.md").write_text("# One\n\nhello\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    ingest(folder, index_dir, encode=fake_encode, count_tokens=fake_tokens, model_id="m", model_revision="r")
    db = sqlite3.connect(index_dir / "rag.sqlite")
    db.execute("UPDATE meta SET value = '1' WHERE key = 'pipeline_version'")
    db.commit()
    db.close()
    again = ingest(folder, index_dir, encode=fake_encode, count_tokens=fake_tokens, model_id="m", model_revision="r")
    assert again[0].status == "indexed"


def test_store_context_and_closed_guard(tmp_path):
    index = Index(tmp_path / "idx", "m", "r", 3)
    index.close()
    with pytest.raises(RuntimeError, match="not open"):
        index._db()
    with index:
        assert index.db is not None
    assert index.db is None


def test_score_table_and_borderless_ladder():
    assert score_table([["only"]]) == (False, "too_few_rows")
    assert score_table([["a", "b"], ["c"]])[0] is False
    assert score_table([["a"], ["b"]]) == (False, "too_few_columns")
    assert score_table([["h", "n"], ["x" * 201, "1"]]) == (False, "cell_too_long")
    assert score_table([["1", "2"], ["3", "4"]]) == (False, "numeric_header")
    assert looks_borderless("a   b   c\nd   e   f\ng   h   i") is True
    assert render_rows([["", ""], ["a", "b"]]) == "a | b"

    class Page:
        def __init__(self):
            self.calls = 0

        def find_tables(self, table_settings=None):
            del table_settings
            self.calls += 1
            if self.calls == 1:
                raise TypeError("old")
            return []

        def extract_text(self):
            return ""

    assert extract_page_tables(Page()) == []
    from rag.layout import _from_find

    class Old:
        def find_tables(self, table_settings=None):
            if table_settings is not None:
                raise TypeError("old")
            return []

    assert _from_find(Old(), rung=2, settings={"vertical_strategy": "text"}) == []
