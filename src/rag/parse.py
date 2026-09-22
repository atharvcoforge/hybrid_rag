import csv
import io
import re
from pathlib import Path

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
            Block(block.kind, block.text, block.heading_path, block.page, start, pos)
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
    blocks: list[Block] = []
    try:
        with pdfplumber.open(path) as pdf:
            page_texts = []
            page_tables: list[list[str]] = []
            for page in pdf.pages:
                found = list(page.find_tables() or [])
                bboxes = [table.bbox for table in found]
                rendered_tables = []
                for table in found:
                    body = _pdf_table(table.extract())
                    if body:
                        rendered_tables.append(body)
                if bboxes:
                    cropped = page.filter(_outside_tables(bboxes))
                    text = cropped.extract_text() or ""
                else:
                    text = page.extract_text() or ""
                page_texts.append(text)
                page_tables.append(rendered_tables)
            cleaned = drop_page_chrome(page_texts)
            for number, (text, tables) in enumerate(zip(cleaned, page_tables), start=1):
                norm = normalize_text(text)
                for paragraph in [part.strip() for part in re.split(r"\n\s*\n", norm) if part.strip()]:
                    blocks.append(_block("prose", paragraph, "", number))
                for rendered in tables:
                    blocks.append(_block("table", rendered, "", number))
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(str(path), "unreadable file") from exc
    if not blocks:
        raise IngestError(str(path), "no text")
    return blocks


def _outside_tables(bboxes: list[tuple[float, float, float, float]]):
    def keep(obj):
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
            if style.startswith("Heading"):
                digits = re.findall(r"\d+", style)
                level = int(digits[0]) if digits else 1
                name = normalize_text(paragraph.text)
                heading = heading[: level - 1]
                if name:
                    heading.append(name)
                continue
            kind = "list" if style.startswith("List") else "prose"
            body = normalize_text(paragraph.text)
            if body:
                blocks.append(_block(kind, body, _path(heading), 0))
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [normalize_text(cell.text) for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            body = "\n".join(rows).strip()
            if body:
                blocks.append(_block("table", body, _path(heading), 0))
    return blocks


def _block(kind: str, text: str, heading_path: str, page: int) -> Block:
    return Block(kind, text, heading_path, page, 0, 0)


def _path(heading: list[str]) -> str:
    return " > ".join(heading)


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


def _pdf_table(table: list) -> str:
    rows = []
    for row in table:
        cells = [normalize_text(cell or "") for cell in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows).strip()
