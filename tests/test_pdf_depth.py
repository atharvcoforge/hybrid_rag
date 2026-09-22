"""PDF page isolation: one bad page must not kill the file (§05)."""

from rag.models import Block
from rag.parse import parse_pdf_pages


class _Page:
    def __init__(self, text="", width=612, height=792, fail=False):
        self._text = text
        self.width = width
        self.height = height
        self._fail = fail

    def extract_text(self):
        if self._fail:
            raise RuntimeError("boom")
        return self._text

    def find_tables(self, table_settings=None):
        return []

    def filter(self, _pred):
        return self

    def images(self):
        return []

    def to_image(self, resolution=150):
        raise RuntimeError("no raster")


def test_page_failure_is_isolated():
    pages = [
        _Page("Healthy page one."),
        _Page(fail=True),
        _Page("Healthy page three."),
    ]
    blocks, errors = parse_pdf_pages(pages)
    assert any("Healthy page one." in b.text for b in blocks)
    assert any("Healthy page three." in b.text for b in blocks)
    assert len(errors) == 1
    assert "page 2" in errors[0]


def test_image_only_page_uses_ocr_engine():
    pages = [_Page("")]  # coverage near zero

    def engine(_image):
        return "Signed on 12 March 2025", 0.88

    blocks, errors = parse_pdf_pages(
        pages,
        ocr_engine=engine,
        page_image=lambda _p: "fake-image",
    )
    assert not errors
    assert any(b.ocr and "12 March 2025" in b.text for b in blocks)
    assert any(b.ocr_confidence == 0.88 for b in blocks)


def test_figure_caption_is_derived():
    pages = [_Page("See Figure 1 below.")]

    def captioner(_image):
        return "Line chart of Scope 1 emissions to 2040"

    blocks, _errors = parse_pdf_pages(
        pages,
        captioner=captioner,
        page_images=lambda _p: ["fig"],
    )
    figures = [b for b in blocks if b.kind == "figure"]
    assert figures
    assert all(b.derived for b in figures)
    assert "Scope 1" in figures[0].text
