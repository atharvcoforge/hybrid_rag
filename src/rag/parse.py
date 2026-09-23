import csv
import io
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag.models import Block, IngestError
from rag.normalize import drop_page_chrome, normalize_text

MIMES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".csv": "text/csv",
}

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TITLE_SMALL = frozenset({"and", "of", "the", "to", "for", "in", "on", "a", "an", "by", "with"})
_FENCE = re.compile(r"^```")
_BULLET = re.compile(r"^(\s*)([-*+]|\d+\.)\s+(.*)$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")


def parse_file(path: Path) -> tuple[str, list[Block]]:
    suffix = path.suffix.lower()
    if suffix not in MIMES:
        raise IngestError(str(path), "unsupported file type")
    try:
        if suffix == ".pdf":
            blocks = parse_pdf(path)
        elif suffix == ".docx":
            blocks = parse_docx(path)
        else:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            parser = {
                ".txt": parse_txt,
                ".md": parse_md,
                ".html": parse_html,
                ".csv": parse_csv,
            }[suffix]
            blocks = parser(text, path.name)
    except IngestError:
        raise
    except OSError as exc:
        raise IngestError(str(path), "unreadable file") from exc
    blocks = [block for block in blocks if block.text.strip()]
    if not blocks:
        raise IngestError(str(path), "no text")
    return MIMES[suffix], assign_offsets(blocks)


def assign_offsets(blocks: list[Block]) -> list[Block]:
    placed = []
    pos = 0
    for i, block in enumerate(blocks):
        if i:
            pos += 1
        start = pos
        pos += len(block.text)
        placed.append(
            Block(
                block.kind,
                block.text,
                block.heading_path,
                block.page,
                start,
                pos,
                derived=block.derived,
                ocr=block.ocr,
                ocr_confidence=block.ocr_confidence,
                flagged=block.flagged,
            )
        )
    return placed


def parse_txt(text: str, filename: str) -> list[Block]:
    heading = Path(filename).stem
    norm = normalize_text(text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", norm) if part.strip()]
    return [_block("prose", paragraph, heading, 0) for paragraph in paragraphs]


def parse_md(text: str, filename: str) -> list[Block]:
    del filename
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    heading: list[str] = []
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line.strip()):
            i += 1
            buf = []
            while i < len(lines) and not _FENCE.match(lines[i].strip()):
                buf.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            code = normalize_text("\n".join(buf), code=True)
            if code:
                blocks.append(_block("code", code, _path(heading), 0))
            continue
        match = _HEADING.match(line)
        if match:
            level = len(match.group(1))
            name = normalize_text(match.group(2))
            heading = heading[: level - 1]
            if name:
                heading.append(name)
            i += 1
            continue
        if _table_at(lines, i):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            table = _markdown_table(table_lines)
            if table:
                blocks.append(_block("table", table, _path(heading), 0))
            continue
        bullet = _BULLET.match(line)
        if bullet:
            item = normalize_text(bullet.group(3))
            if item:
                blocks.append(_block("list", item, _path(heading), 0))
            i += 1
            continue
        if not line.strip():
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines) and _continues_paragraph(lines, i):
            para.append(lines[i])
            i += 1
        body = normalize_text("\n".join(para))
        if body:
            blocks.append(_block("prose", body, _path(heading), 0))
    return blocks


def parse_html(text: str, filename: str) -> list[Block]:
    del filename
    from rag.html_parse import HTMLText

    parser = HTMLText()
    parser.feed(text)
    parser.close()
    return [_block(kind, body, path, 0) for kind, body, path in parser.blocks]


def parse_csv(text: str, filename: str) -> list[Block]:
    heading = Path(filename).stem
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header = rows[0]
    blocks = []
    data = rows[1:] if len(rows) > 1 else []
    if not data and any(cell.strip() for cell in header):
        data = [header]
        header = []
    for row in data:
        parts = []
        for name, cell in zip(header, row):
            if cell.strip():
                label = name.strip() or "col"
                parts.append(f"{label}: {cell.strip()}")
        if len(row) > len(header):
            parts.extend(cell.strip() for cell in row[len(header) :] if cell.strip())
        body = normalize_text("; ".join(parts))
        if body:
            blocks.append(_block("row", body, heading, 0))
    return blocks


def parse_pdf(path: Path) -> list[Block]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise IngestError(str(path), "pdfplumber is not installed") from exc
    try:
        with pdfplumber.open(path) as pdf:
            blocks, errors = parse_pdf_pages(list(pdf.pages))
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(str(path), "unreadable file") from exc
    if not blocks:
        raise IngestError(str(path), "no text")
    for message in errors:
        blocks.append(_block("prose", f"[page error] {message}", "", 0, flagged=True))
    return blocks


def parse_pdf_pages(
    pages: list[Any],
    *,
    layout_extract: Callable[..., Any] | None = None,
    ocr_engine: Callable[[Any], tuple[str, float]] | None = None,
    captioner: Callable[[Any], str] | None = None,
    page_image: Callable[[Any], Any] | None = None,
    page_images: Callable[[Any], list[Any]] | None = None,
) -> tuple[list[Block], list[str]]:
    """Per-page PDF parse. One broken page becomes an error entry, not a hard fail."""
    from rag.layout import extract_page_tables
    from rag.ocr import caption_figure, is_image_only, ocr_page
    from rag.ocr import page_image as default_page_image

    if page_image is None:
        page_image = default_page_image
    if layout_extract is None:
        layout_extract = _docling_extract

    page_texts: list[str] = []
    page_tables: list[list[Any]] = []
    page_extras: list[list[Block]] = []
    errors: list[str] = []

    for number, page in enumerate(pages, start=1):
        try:
            tables = extract_page_tables(page, layout_extract=layout_extract)
            bboxes = [t.bbox for t in tables if t.bbox]
            if bboxes:
                cropped = page.filter(_outside_tables(bboxes))
                text = cropped.extract_text() or ""
            else:
                text = page.extract_text() or ""

            extras: list[Block] = []
            if is_image_only(page) and not (text or "").strip() and not tables:
                image = page_image(page)
                if image is not None or ocr_engine is not None:
                    result = ocr_page(image if image is not None else page, engine=ocr_engine)
                    if result.text:
                        extras.append(
                            _block(
                                "prose",
                                normalize_text(result.text),
                                "",
                                number,
                                ocr=True,
                                ocr_confidence=result.confidence,
                            )
                        )

            images = page_images(page) if page_images is not None else _embedded_images(page)
            for image in images or []:
                caption = caption_figure(image, captioner=captioner)
                if caption:
                    extras.append(
                        _block("figure", normalize_text(caption), "", number, derived=True)
                    )

            page_texts.append(text)
            page_tables.append(tables)
            page_extras.append(extras)
        except Exception as exc:  # noqa: BLE001 — parser libraries raise an open set
            page_texts.append("")
            page_tables.append([])
            page_extras.append([])
            errors.append(f"page {number}: {exc}")

    cleaned = drop_page_chrome(page_texts)
    blocks: list[Block] = []
    for number, (text, tables, extras) in enumerate(
        zip(cleaned, page_tables, page_extras), start=1
    ):
        norm = normalize_text(text)
        blocks.extend(pdf_prose_blocks(norm, number))
        for table in tables:
            blocks.append(_block("table", table.text, "", number, flagged=table.flagged))
        blocks.extend(extras)
    return blocks, errors


def _outside_tables(
    bboxes: list[tuple[float, float, float, float]],
) -> Callable[[Any], bool]:
    def keep(obj: Any) -> bool:
        x0 = obj.get("x0")
        x1 = obj.get("x1")
        top = obj.get("top")
        bottom = obj.get("bottom")
        if None in (x0, x1, top, bottom):
            return True
        for bx0, btop, bx1, bbottom in bboxes:
            if x0 >= bx0 - 1 and x1 <= bx1 + 1 and top >= btop - 1 and bottom <= bbottom + 1:
                return False
        return True

    return keep


def _docling_extract(page: Any) -> None:
    """Optional Docling rung. Absent package ⇒ no escalation."""
    try:
        import docling  # noqa: F401
    except ImportError:
        return
    del page
    return


def _embedded_images(page: Any) -> list[Any]:
    try:
        return list(getattr(page, "images", []) or [])
    except Exception:  # noqa: BLE001 — parser libraries raise an open set
        return []


def parse_docx(path: Path) -> list[Block]:
    try:
        from docx import Document
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise IngestError(str(path), "python-docx is not installed") from exc
    try:
        document = Document(str(path))
    except Exception as exc:
        raise IngestError(str(path), "unreadable file") from exc
    heading: list[str] = []
    blocks: list[Block] = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            style = paragraph.style.name if paragraph.style is not None else ""
            for box_text in _textbox_texts(child):
                body = normalize_text(box_text)
                if body:
                    blocks.append(_block("prose", body, _path(heading), 0))
            if style.startswith("Heading"):
                digits = re.findall(r"\d+", style)
                level = int(digits[0]) if digits else 1
                name = normalize_text(_docx_text(child))
                heading = heading[: level - 1]
                if name:
                    heading.append(name)
                continue
            kind = "list" if style.startswith("List") else "prose"
            body = normalize_text(_docx_text(child))
            if body:
                blocks.append(_block(kind, body, _path(heading), 0))
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [normalize_text(_docx_text(cell._tc)) for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            body = "\n".join(rows).strip()
            if body:
                blocks.append(_block("table", body, _path(heading), 0))
    for note in _footnote_texts(document):
        body = normalize_text(note)
        if body:
            blocks.append(_block("prose", body, _path(heading) or "Footnotes", 0))
    return blocks


def _docx_text(element: Any, *, include_textboxes: bool = False) -> str:
    """Keep insertions, drop tracked deletions."""
    from docx.oxml.ns import qn

    parts: list[str] = []
    for node in element.iter():
        tag = node.tag
        if tag in (qn("w:del"), qn("w:delText")):
            continue
        if tag == qn("w:t") and _inside_deletion(node):
            continue
        if tag == qn("w:t") and not include_textboxes and _inside_textbox(node):
            continue
        if tag == qn("w:t") and node.text:
            parts.append(node.text)
        if tag == qn("w:tab"):
            parts.append("\t")
    return "".join(parts)


def _inside_deletion(node: Any) -> bool:
    from docx.oxml.ns import qn

    parent = node.getparent()
    while parent is not None:
        if parent.tag == qn("w:del"):
            return True
        parent = parent.getparent()
    return False


def _inside_textbox(node: Any) -> bool:
    from docx.oxml.ns import qn

    parent = node.getparent()
    while parent is not None:
        if parent.tag == qn("w:txbxContent"):
            return True
        parent = parent.getparent()
    return False


def _textbox_texts(element: Any) -> list[str]:
    from docx.oxml.ns import qn

    out = []
    for node in element.iter():
        if node.tag == qn("w:txbxContent"):
            text = _docx_text(node, include_textboxes=True)
            if text.strip():
                out.append(text)
    return out


def _footnote_texts(document: Any) -> list[str]:
    from docx.oxml.ns import qn
    from lxml import etree

    out: list[str] = []
    try:
        package = document.part.package
    except Exception:  # noqa: BLE001 — parser libraries raise an open set
        return out
    for other in package.parts:
        name = getattr(other, "partname", None)
        if name is None or "footnotes" not in str(name):
            continue
        root = getattr(other, "element", None)
        if root is None:
            blob = getattr(other, "blob", None)
            if not blob:
                continue
            root = etree.fromstring(blob)
        for footnote in root.iter():
            if footnote.tag != qn("w:footnote"):
                continue
            fid = footnote.get(qn("w:id"))
            if fid in ("-1", "0"):
                continue
            text = _docx_text(footnote, include_textboxes=True)
            if text.strip():
                out.append(text)
    return out


def _block(
    kind: str,
    text: str,
    heading_path: str,
    page: int,
    *,
    derived: bool = False,
    ocr: bool = False,
    ocr_confidence: float | None = None,
    flagged: bool = False,
) -> Block:
    if kind == "figure":
        derived = True
    return Block(
        kind,
        text,
        heading_path,
        page,
        0,
        0,
        derived=derived,
        ocr=ocr,
        ocr_confidence=ocr_confidence,
        flagged=flagged,
    )


def _path(heading: list[str]) -> str:
    return " > ".join(heading)


def is_section_title(line: str) -> bool:
    """PDF section title: short title-case line, not a sentence, bullet, or table row."""
    text = line.strip()
    if not text or len(text) < 3 or len(text) > 70:
        return False
    if text[0] in "•-*|":
        return False
    if re.match(r"^\d+\.", text):
        return False
    if text[-1] in ".!?,;:":
        return False
    if any(char.isdigit() for char in text) or "," in text or "|" in text:
        return False
    words = text.split()
    if not 2 <= len(words) <= 8 or not text[0].isupper():
        return False
    for word in words:
        bare = word.strip("()'\"")
        if not bare:
            return False
        if bare.casefold() in _TITLE_SMALL:
            continue
        if not bare[0].isupper():
            return False
    return True


def pdf_prose_blocks(text: str, page: int) -> list[Block]:
    """Split PDF prose on section titles so each section can be its own parent."""
    heading: list[str] = []
    blocks: list[Block] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if body:
            blocks.append(_block("prose", body, _path(heading), page))

    lines = (text or "").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if is_section_title(line):
            # A name followed by a role is one block, not two sections.
            run = [line.strip()]
            nxt = index + 1
            while nxt < len(lines) and is_section_title(lines[nxt]):
                run.append(lines[nxt].strip())
                nxt += 1
            flush()
            heading = [run[0]]
            blocks.append(_block("prose", "\n".join(run), _path(heading), page))
            index = nxt
            continue
        if not line.strip():
            flush()
            index += 1
            continue
        buf.append(line.strip())
        index += 1
    flush()
    return blocks


def _continues_paragraph(lines: list[str], i: int) -> bool:
    line = lines[i]
    if not line.strip():
        return False
    if _HEADING.match(line) or _FENCE.match(line.strip()) or _BULLET.match(line):
        return False
    return not _table_at(lines, i)


def _table_at(lines: list[str], i: int) -> bool:
    if i + 1 >= len(lines) or "|" not in lines[i]:
        return False
    return _TABLE_SEP.match(lines[i + 1].strip()) is not None


def _markdown_table(lines: list[str]) -> str:
    rows = []
    for i, line in enumerate(lines):
        if i == 1 and _TABLE_SEP.match(line.strip()):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if any(cells):
            rows.append(" | ".join(cells))
    return normalize_text("\n".join(rows))


def _pdf_table(table: list[Any]) -> str:
    rows = []
    for row in table:
        cells = [normalize_text(cell or "") for cell in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows).strip()
