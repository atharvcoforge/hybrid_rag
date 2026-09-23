"""OCR and figure captions for image-only PDF pages (§05, F-13).

RapidOCR and the VLM captioner are injectable so tests never need weights.
Blocks produced here carry ocr/derived flags for the citation gates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Characters per square point below which a page is treated as image-only.
_COVERAGE_FLOOR = 0.002


@dataclass
class OcrResult:
    text: str
    confidence: float


def text_coverage(page: Any) -> float:
    """Characters extracted / page area. Low ⇒ scan or figure page."""
    try:
        text = page.extract_text() or ""
    except Exception:  # noqa: BLE001 — pdfplumber/OCR raises an open set
        text = ""
    width = float(getattr(page, "width", 0) or 0)
    height = float(getattr(page, "height", 0) or 0)
    area = width * height
    if area <= 0:
        return 0.0
    return len(text.strip()) / area


def is_image_only(page: Any, *, floor: float = _COVERAGE_FLOOR) -> bool:
    return text_coverage(page) < floor


def ocr_page(
    image: Any, *, engine: Callable[[Any], tuple[str, float]] | None = None
) -> OcrResult:
    """Run OCR on a page image. `engine(image) -> (text, confidence)`."""
    if engine is None:
        engine = _default_ocr_engine()
    if engine is None:
        return OcrResult("", 0.0)
    text, confidence = engine(image)
    return OcrResult(text=(text or "").strip(), confidence=float(confidence or 0.0))


def caption_figure(image: Any, *, captioner: Callable[[Any], str] | None = None) -> str:
    """One-sentence VLM description. Empty when no captioner is available."""
    if captioner is None:
        captioner = _default_captioner()
    if captioner is None:
        return ""
    try:
        return (captioner(image) or "").strip()
    except Exception:  # noqa: BLE001 — pdfplumber/OCR raises an open set
        return ""


def page_image(page: Any) -> Any:
    """Best-effort raster of a pdfplumber page. None when rendering fails."""
    try:
        if hasattr(page, "to_image"):
            return page.to_image(resolution=150).original
    except Exception:  # noqa: BLE001 — pdfplumber/OCR raises an open set
        return None
    return None


def _default_ocr_engine() -> Callable[[Any], tuple[str, float]] | None:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    reader = RapidOCR()

    def run(image: Any) -> tuple[str, float]:
        result, _ = reader(image)
        if not result:
            return "", 0.0
        lines: list[str] = []
        scores: list[float] = []
        for item in result:
            # RapidOCR: [box, text, score]
            if len(item) >= 3:
                lines.append(str(item[1]))
                scores.append(float(item[2]))
        conf = sum(scores) / len(scores) if scores else 0.0
        return "\n".join(lines), conf

    return run


def _default_captioner() -> Callable[[Any], str] | None:
    # Optional local VLM via llama.cpp multimodal — absent ⇒ skip quietly.
    return None
