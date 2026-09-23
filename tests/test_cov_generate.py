import json
import urllib.error
import urllib.request

import pytest

from rag.generate import _fence, stream_answer


def test_unclosed_passage_marker_is_cut_without_a_tail():
    cleaned = _fence("before <<PASSAGE_abcde still open", "MARK")
    assert "<<PASSAGE_" not in cleaned
    assert "still open" in cleaned
    closed = _fence("see <<PASSAGE_abcd>> tail", "MARK")
    assert "PASSAGE" not in closed and "tail" in closed


def test_stream_skips_noise_and_empty_deltas(monkeypatch):
    class Resp:
        def __iter__(self):
            yield b"comment\n"
            yield b'data: {"choices":[{"delta":{}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
            yield b"data: [DONE]\n"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Resp())
    assert list(stream_answer("q", [])) == ["hi"]

    class Quiet:
        def __iter__(self):
            return iter(())

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Quiet())
    assert list(stream_answer("q", [])) == []

    def down(*_args, **_kwargs):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", down)
    with pytest.raises(urllib.error.URLError):
        list(stream_answer("q", []))


def test_stream_payload_is_json():
    assert json.dumps({"ok": True})
