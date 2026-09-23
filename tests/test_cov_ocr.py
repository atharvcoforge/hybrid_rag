import sys
import types

from rag.ocr import ocr_page


def test_rapidocr_empty_and_short_rows(monkeypatch):
    calls = {"n": 0}

    class Reader:
        def __call__(self, _image):
            calls["n"] += 1
            if calls["n"] == 1:
                return None, None
            return [(["box"],), (["box"], "line", 0.5)], None

    module = types.ModuleType("rapidocr_onnxruntime")
    module.RapidOCR = Reader
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", module)
    empty = ocr_page(object(), engine=None)
    assert empty.text == "" and empty.confidence == 0.0
    found = ocr_page(object(), engine=None)
    assert found.text == "line"
    assert found.confidence == 0.5
