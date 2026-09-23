"""PDF table extraction: ruled → borderless → layout model (§05).

Every table is scored before indexing. Failures escalate; a table that still
fails is emitted as flagged prose so the ingest report can surface it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_NUMERIC = re.compile(
    r"""^
    [\$£€]?\s*
    -?
    (?:\d{1,3}(?:,\d{3})+|\d+)
    (?:\.\d+)?
    %?
    $""",
    re.VERBOSE,
)

_CELL_MAX = 200


@dataclass
class ExtractedTable:
    text: str
    rows: list[list[str]]
    rung: int
    ok: bool
    reason: str = ""
    flagged: bool = False
    bbox: tuple[float, float, float, float] | None = None


def score_table(rows: list[list[str]]) -> tuple[bool, str]:
    """Structure check used both as the escalate trigger and as the ingest flag."""
    cleaned = [[(cell or "").strip() for cell in row] for row in rows if any((cell or "").strip() for cell in row)]
    if len(cleaned) < 2:
        return False, "too_few_rows"
    widths = {len(row) for row in cleaned}
    if len(widths) != 1:
        return False, "ragged_columns"
    width = cleaned[0].__len__()
    if width < 2:
        return False, "too_few_columns"
    for row in cleaned:
        for cell in row:
            if len(cell) > _CELL_MAX:
                return False, "cell_too_long"
    header = cleaned[0]
    if all(_is_numeric(cell) for cell in header if cell):
        return False, "numeric_header"
    # Columns whose header looks numeric-unit-ish (digits absent, but data numeric)
    # must actually hold numbers where cells are non-empty.
    for col in range(width):
        header_cell = header[col]
        if not header_cell or _is_numeric(header_cell):
            continue
        values = [row[col] for row in cleaned[1:] if col < len(row) and row[col]]
        if not values:
            continue
        numeric_hits = sum(1 for value in values if _is_numeric(value))
        # If most cells look numeric, demand they all do — catches "fourteen" under tCO2e.
        if numeric_hits >= max(1, len(values) // 2) and numeric_hits < len(values):
            return False, "non_numeric_cell"
        if _header_implies_numeric(header_cell) and numeric_hits < len(values):
            return False, "non_numeric_cell"
    return True, ""


def looks_borderless(text: str) -> bool:
    """≥3 lines with ≥3 whitespace-aligned columns → worth trying text strategy."""
    hits = 0
    for line in (text or "").splitlines():
        if _aligned_columns(line) >= 3:
            hits += 1
            if hits >= 3:
                return True
    return False


def extract_page_tables(page: Any, *, layout_extract: Callable[..., Any] | None = None) -> list[ExtractedTable]:
    """Three-rung ladder. `layout_extract(page) -> list[list[list[str]]]` is Docling."""
    ruled = _from_find(page, rung=1, settings=None)
    if ruled:
        return _finalize(ruled, page, layout_extract)

    text = ""
    try:
        text = page.extract_text() or ""
    except Exception:  # noqa: BLE001 — pdfplumber raises an open set
        text = ""
    if not looks_borderless(text):
        return []

    settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
    }
    borderless = _from_find(page, rung=2, settings=settings)
    if borderless:
        return _finalize(borderless, page, layout_extract)
    return []


def render_rows(rows: list[list[str]]) -> str:
    lines = []
    for row in rows:
        cells = [(cell or "").strip() for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines).strip()


def _finalize(candidates: list[ExtractedTable], page: Any, layout_extract: Callable[..., Any] | None) -> list[ExtractedTable]:
    out: list[ExtractedTable] = []
    for table in candidates:
        ok, reason = score_table(table.rows)
        if ok:
            table.ok = True
            table.reason = ""
            table.flagged = False
            out.append(table)
            continue
        escalated = _escalate(page, layout_extract, bbox=table.bbox)
        if escalated:
            out.extend(escalated)
            continue
        table.ok = False
        table.reason = reason
        table.flagged = True
        out.append(table)
    return out


def _escalate(page: Any, layout_extract: Callable[..., Any] | None, *, bbox: Any) -> list[ExtractedTable]:
    if layout_extract is None:
        return []
    try:
        raw = layout_extract(page) or []
    except Exception:  # noqa: BLE001 — pdfplumber raises an open set
        return []
    tables = []
    for rows in raw:
        body = render_rows(rows)
        if not body:
            continue
        ok, reason = score_table(rows)
        tables.append(
            ExtractedTable(
                text=body,
                rows=rows,
                rung=3,
                ok=ok,
                reason="" if ok else reason,
                flagged=not ok,
                bbox=bbox,
            )
        )
    return tables


def _from_find(page: Any, *, rung: int, settings: dict[str, str] | None) -> list[ExtractedTable]:
    try:
        found = list(page.find_tables(table_settings=settings) if settings else page.find_tables() or [])
    except TypeError:
        # Older pdfplumber: find_tables() takes no kwargs.
        if settings:
            return []
        found = list(page.find_tables() or [])
    except Exception:  # noqa: BLE001 — pdfplumber raises an open set
        return []
    tables = []
    for item in found:
        try:
            rows = item.extract()
        except Exception:  # noqa: BLE001, S112 — one bad table must not drop the page
            continue
        if not rows:
            continue
        body = render_rows(rows)
        if not body:
            continue
        bbox = getattr(item, "bbox", None)
        tables.append(
            ExtractedTable(text=body, rows=rows, rung=rung, ok=True, bbox=bbox)
        )
    return tables


def _is_numeric(cell: str) -> bool:
    text = (cell or "").strip().replace("\u2212", "-")
    if not text:
        return False
    return _NUMERIC.match(text) is not None


def _header_implies_numeric(header: str) -> bool:
    lower = header.casefold()
    markers = ("tco2", "kwp", "%", "percent", "total", "scope", "fy", "year", "amount", "emissions")
    return any(marker in lower for marker in markers)


def _aligned_columns(line: str) -> int:
    # Split on 2+ spaces — the usual whitespace-column pattern in policy PDFs.
    parts = [part for part in re.split(r" {2,}", line.strip()) if part]
    return len(parts)
