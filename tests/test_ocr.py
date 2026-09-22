"""OCR coverage heuristic and injectable engines (§05, F-13)."""

from rag.ocr import caption_figure, is_image_only, ocr_page, text_coverage


class _Page:
    def __init__(self, text, width=612, height=792):
        self._text = text
        self.width = width
        self.height = height

    def extract_text(self):
        return self._text


def test_text_coverage_is_chars_over_area():
    page = _Page("hello", width=100, height=100)
    assert text_coverage(page) == 5 / 10_000


def test_image_only_when_coverage_is_near_zero():
    assert is_image_only(_Page(""))
    assert is_image_only(_Page("x"))
    assert not is_image_only(_Page("A" * 5000))


def test_ocr_page_uses_injected_engine():
    result = ocr_page("img", engine=lambda _img: ("Signed by Alice", 0.91))
    assert result.text == "Signed by Alice"
    assert result.confidence == 0.91


def test_ocr_page_without_engine_returns_empty():
    result = ocr_page("img", engine=None)
    # No RapidOCR installed in the test env → empty, not an exception.
    assert result.text == ""
    assert result.confidence == 0.0


def test_caption_figure_marks_nothing_when_captioner_missing():
    assert caption_figure("img", captioner=None) == ""


def test_caption_figure_uses_injected_captioner():
    assert caption_figure("img", captioner=lambda _img: "Bar chart of FY24 emissions") == (
        "Bar chart of FY24 emissions"
    )
