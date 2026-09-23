"""HTML extraction with a proper element stack (F-11)."""

from __future__ import annotations

from html.parser import HTMLParser

from rag.normalize import normalize_text


def _path(heading: list[str]) -> str:
    return " > ".join(heading)


_BLOCK = frozenset(
    {
        "div",
        "section",
        "article",
        "blockquote",
        "dl",
        "figcaption",
        "aside",
        "main",
        "header",
        "footer",
        "nav",
    }
)
_HEADINGS = frozenset({f"h{n}" for n in range(1, 7)})
_SKIP = frozenset({"script", "style", "noscript"})


class HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.heading: list[str] = []
        self.stack: list[str] = []
        self.buf: list[str] = []
        self.blocks: list[tuple[str, str, str]] = []
        self.rows: list[str] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.in_pre = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        self.stack.append(tag)
        if tag in _HEADINGS:
            self._flush_prose()
            self.buf = []
        elif tag == "pre":
            self._flush_prose()
            self.in_pre = True
            self.buf = []
        elif tag in ("p", "li") or tag in _BLOCK:
            if tag != "li" or "table" not in self.stack[:-1]:
                self._flush_prose()
            self.buf = []
        elif tag == "table":
            self._flush_prose()
            self.rows = []
        elif tag == "tr" and self.rows is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag == "img":
            alt = dict(attrs).get("alt") or dict(attrs).get("ALT") or ""
            alt = normalize_text(alt)
            if alt:
                self.blocks.append(("caption", alt, _path(self.heading)))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag in ("td", "th") and self.cell is not None:
            if self.row is not None:
                self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if any(self.row) and self.rows is not None:
                self.rows.append(" | ".join(self.row))
            self.row = None
        elif tag == "table" and self.rows is not None:
            text = normalize_text("\n".join(self.rows))
            if text:
                self.blocks.append(("table", text, _path(self.heading)))
            self.rows = None
        elif tag in _HEADINGS:
            name = normalize_text("".join(self.buf))
            level = int(tag[1])
            self.heading = self.heading[: level - 1]
            if name:
                self.heading.append(name)
            self.buf = []
        elif tag == "pre":
            text = normalize_text("".join(self.buf), code=True)
            if text:
                self.blocks.append(("code", text, _path(self.heading)))
            self.in_pre = False
            self.buf = []
        elif tag in ("p", "li") or tag in _BLOCK:
            kind = "list" if tag == "li" else "prose"
            text = normalize_text("".join(self.buf))
            if text:
                self.blocks.append((kind, text, _path(self.heading)))
            self.buf = []
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.stack.pop()
            if self.stack:
                self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.rows is not None and self.row is None:
            return
        else:
            self.buf.append(data)

    def close(self) -> None:
        self._flush_prose()
        super().close()

    def _flush_prose(self) -> None:
        text = normalize_text("".join(self.buf))
        if text:
            self.blocks.append(("prose", text, _path(self.heading)))
        self.buf = []
