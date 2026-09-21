import re
import unicodedata

_HYPHEN = re.compile(r"([A-Za-z])-\n\s*([a-z])")


def normalize_text(text: str, *, code: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not code:
        text = _HYPHEN.sub(r"\1\2", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def drop_page_chrome(pages: list[str]) -> list[str]:
    """Drop a short line that shows up on more than half the pages."""
    if len(pages) < 2:
        return pages
    split = []
    for page in pages:
        split.append([ln.strip() for ln in page.splitlines() if ln.strip()])
    counts: dict[str, int] = {}
    for lines in split:
        for line in set(lines):
            counts[line] = counts.get(line, 0) + 1
    banned = {line for line, n in counts.items() if n > len(pages) / 2 and len(line) < 80}
    if not banned:
        return pages
    cleaned = []
    for lines in split:
        cleaned.append("\n".join(line for line in lines if line not in banned))
    return cleaned
