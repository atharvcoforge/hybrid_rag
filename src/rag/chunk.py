import hashlib
import re

from rag.models import (
    CHILD_MAX,
    CHILD_OVERLAP,
    CHILD_TARGET,
    PARENT_MAX,
    PARENT_OVERLAP,
    PARENT_TARGET,
    PIPELINE_VERSION,
    Block,
    Child,
    Parent,
    make_embed_text,
)

_SENTENCE = re.compile(r"[.!?]\s+|\n+")


def chunk_document(
    doc_id: str, blocks: list[Block], count_tokens, *, context_fn=None
) -> tuple[list[Parent], list[Child]]:
    parents: list[Parent] = []
    children: list[Child] = []
    for section in _sections(blocks):
        for parent in _pack_section(doc_id, section, count_tokens, len(parents)):
            parents.append(parent[0])
            children.extend(parent[1])
    children = _merge_tiny(children, count_tokens)
    children = _dedupe(children)
    if context_fn is not None:
        _apply_context(children, parents, context_fn, count_tokens)
    kept_parents = {child.parent_id for child in children}
    parents = [parent for parent in parents if parent.parent_id in kept_parents]
    for index, parent in enumerate(parents):
        parent.parent_index = index
    by_id = {parent.parent_id: parent.parent_index for parent in parents}
    for index, child in enumerate(children):
        child.child_index = index
        child.parent_index = by_id[child.parent_id]
    return parents, children


def _sections(blocks: list[Block]) -> list[list[Block]]:
    groups: list[list[Block]] = []
    for block in blocks:
        if not block.text.strip():
            continue
        if block.kind in ("table", "code", "figure"):
            groups.append([block])
            continue
        if (
            groups
            and groups[-1][0].kind not in ("table", "code", "figure")
            and groups[-1][0].heading_path == block.heading_path
            and groups[-1][0].kind != "row"
            and block.kind != "row"
        ):
            groups[-1].append(block)
        elif (
            groups
            and block.kind == "row"
            and groups[-1][0].kind == "row"
            and groups[-1][0].heading_path == block.heading_path
        ):
            groups[-1].append(block)
        else:
            groups.append([block])
    return groups


def _pack_section(doc_id, blocks, count_tokens, parent_index):
    kind = _section_kind(blocks)
    text = "\n".join(block.text for block in blocks)
    start = blocks[0].start_char
    pages = [block.page for block in blocks]
    derived = any(block.derived or block.kind == "figure" for block in blocks)
    if kind in ("table", "code", "figure"):
        parent, _kids = _make_parent(
            doc_id,
            text,
            start,
            blocks[0].heading_path,
            kind,
            pages,
            parent_index,
            count_tokens,
            derived=derived,
        )
        return [(parent, _children_for(parent, kind, count_tokens))]
    if kind == "row":
        windows = _windows(
            text,
            _line_spans(text),
            count_tokens,
            PARENT_TARGET,
            PARENT_MAX,
            0,
            False,
        )
        return _parents_from_windows(
            doc_id,
            text,
            start,
            blocks[0].heading_path,
            kind,
            pages,
            parent_index,
            count_tokens,
            windows,
            derived=derived,
        )
    total = count_tokens(text)
    if total <= PARENT_MAX:
        parent, _ = _make_parent(
            doc_id,
            text,
            start,
            blocks[0].heading_path,
            kind,
            pages,
            parent_index,
            count_tokens,
            derived=derived,
        )
        return [(parent, _children_for(parent, kind, count_tokens))]
    windows = _windows(
        text,
        _piece_spans(text, count_tokens, PARENT_MAX),
        count_tokens,
        PARENT_TARGET,
        PARENT_MAX,
        PARENT_OVERLAP,
        True,
    )
    return _parents_from_windows(
        doc_id,
        text,
        start,
        blocks[0].heading_path,
        kind,
        pages,
        parent_index,
        count_tokens,
        windows,
        derived=derived,
    )


def _parents_from_windows(
    doc_id, text, start, heading, kind, pages, parent_index, count_tokens, windows, *, derived=False
):
    packed = []
    for offset, (local_start, local_end) in enumerate(windows):
        raw = text[local_start:local_end]
        body = raw.strip()
        if not body:
            continue
        lead = len(raw) - len(raw.lstrip())
        parent, _kids = _make_parent(
            doc_id,
            body,
            start + local_start + lead,
            heading,
            kind,
            pages,
            parent_index + offset,
            count_tokens,
            derived=derived,
        )
        packed.append((parent, _children_for(parent, kind, count_tokens)))
    return packed


def _children_for(parent: Parent, kind: str, count_tokens) -> list[Child]:
    if kind == "table":
        pieces = [
            _child(parent, body, parent.start_char + row_start, parent.start_char + row_end, count_tokens)
            for body, row_start, row_end in _table_pieces(parent.text, count_tokens)
        ]
        summary = _table_summary_child(parent, count_tokens)
        return ([summary] if summary else []) + pieces
    if kind == "figure":
        return [_child(parent, parent.text, parent.start_char, parent.end_char, count_tokens)]
    if kind in ("code", "row"):
        spans = _windows(
            parent.text,
            _line_spans(parent.text),
            count_tokens,
            CHILD_TARGET,
            CHILD_MAX,
            CHILD_OVERLAP if kind == "code" else 0,
            count_tokens(parent.text) > CHILD_MAX,
        )
    elif count_tokens(parent.text) <= CHILD_MAX:
        spans = [(0, len(parent.text))]
    else:
        spans = _windows(
            parent.text,
            _sentence_spans(parent.text),
            count_tokens,
            CHILD_TARGET,
            CHILD_MAX,
            CHILD_OVERLAP,
            True,
        )
    children = []
    for local_start, local_end in spans:
        raw = parent.text[local_start:local_end]
        body = raw.strip()
        if not body:
            continue
        lead = len(raw) - len(raw.lstrip())
        start = parent.start_char + local_start + lead
        children.append(_child(parent, body, start, start + len(body), count_tokens))
    return children


def _table_summary_child(parent: Parent, count_tokens) -> Child | None:
    lines = [line.strip() for line in parent.text.splitlines() if line.strip()]
    if not lines:
        return None
    header = lines[0]
    cols = [part.strip() for part in header.split("|") if part.strip()]
    units = [col for col in cols if _looks_like_unit(col)]
    parts = []
    if parent.heading_path:
        parts.append(parent.heading_path)
    parts.append("columns: " + ", ".join(cols))
    if units:
        parts.append("units: " + ", ".join(units))
    body = ". ".join(parts)
    child = _child(parent, body, parent.start_char, parent.start_char + len(body), count_tokens)
    child.block_type = "table_summary"
    return child


def _looks_like_unit(header: str) -> bool:
    lower = header.casefold()
    return any(marker in lower for marker in ("tco2", "kwp", "%", "fy", "scope"))


def _label_row(header: str, row: str) -> str:
    heads = [part.strip() for part in header.split("|")]
    cells = [part.strip() for part in row.split("|")]
    parts = []
    for name, cell in zip(heads, cells):
        if cell:
            label = name or "col"
            parts.append(f"{label}: {cell}")
    if len(cells) > len(heads):
        parts.extend(cell for cell in cells[len(heads) :] if cell)
    return "; ".join(parts)


def _child(parent: Parent, body: str, start: int, end: int, count_tokens) -> Child:
    embed_text = make_embed_text(parent.heading_path, body)
    return Child(
        chunk_id=_content_id(str(PIPELINE_VERSION), parent.doc_id, embed_text),
        parent_id=parent.parent_id,
        doc_id=parent.doc_id,
        text=body,
        embed_text=embed_text,
        heading_path=parent.heading_path,
        block_type=parent.block_type,
        start_char=start,
        end_char=end,
        page_start=parent.page_start,
        page_end=parent.page_end,
        child_index=0,
        parent_index=parent.parent_index,
        token_count=count_tokens(embed_text),
        derived=parent.derived,
    )


def _table_pieces(text: str, count_tokens) -> list[tuple[str, int, int]]:
    lines = _line_spans(text)
    if not lines:
        return []
    header_start, header_end = lines[0]
    header = text[header_start:header_end]
    data = lines[1:]
    if not data:
        body = header.strip()
        return [(body, header_start, header_start + len(body))] if body else []
    groups: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    tokens = count_tokens(header)
    for line in data:
        n = count_tokens(text[line[0] : line[1]])
        if current and tokens + n > CHILD_MAX:
            groups.append(current)
            current = []
            tokens = count_tokens(header)
        current.append(line)
        tokens += n
    if current:
        groups.append(current)
    pieces = []
    for group in groups:
        row_start = group[0][0]
        row_end = group[-1][1]
        row_lines = [text[start:end].strip() for start, end in group]
        labeled = [_label_row(header, row) for row in row_lines if row]
        labeled_block = "\n".join(part for part in labeled if part)
        if row_start == data[0][0]:
            body = f"{header.strip()}\n{labeled_block}".strip()
            pieces.append((body, 0, len(body)))
            continue
        body = f"{header.strip()}\n{labeled_block}".strip() if labeled_block else header.strip()
        pieces.append((body, row_start, row_end))
    return pieces


_IDENT = re.compile(r"\d|[A-Za-z]+-\d|\b[A-Z]{2,}-\d")
_MIN_CHILD_TOKENS = 24


def _merge_tiny(children: list[Child], count_tokens) -> list[Child]:
    if len(children) < 2:
        return children
    out: list[Child] = []
    for child in children:
        if (
            out
            and child.parent_id == out[-1].parent_id
            and child.block_type == out[-1].block_type
            and child.block_type not in ("table", "table_summary", "code", "figure")
            and count_tokens(child.text) < _MIN_CHILD_TOKENS
            and not _IDENT.search(child.text)
        ):
            prev = out[-1]
            merged_text = f"{prev.text}\n{child.text}".strip()
            embed = make_embed_text(prev.heading_path, merged_text)
            out[-1] = Child(
                chunk_id=_content_id(str(PIPELINE_VERSION), prev.doc_id, embed),
                parent_id=prev.parent_id,
                doc_id=prev.doc_id,
                text=merged_text,
                embed_text=embed,
                heading_path=prev.heading_path,
                block_type=prev.block_type,
                start_char=prev.start_char,
                end_char=child.end_char,
                page_start=prev.page_start,
                page_end=child.page_end,
                child_index=prev.child_index,
                parent_index=prev.parent_index,
                token_count=count_tokens(embed),
                derived=prev.derived or child.derived,
                context_prefix=prev.context_prefix,
            )
        else:
            out.append(child)
    return out


def _apply_context(children, parents, context_fn, count_tokens) -> None:
    by_id = {parent.parent_id: parent for parent in parents}
    for child in children:
        parent = by_id.get(child.parent_id)
        if parent is None:
            continue
        try:
            prefix = (context_fn(parent.text, child.text) or "").strip()
        except Exception:
            prefix = ""
        if not prefix:
            continue
        child.context_prefix = prefix
        child.embed_text = f"{prefix}\n{child.embed_text}"
        child.token_count = count_tokens(child.embed_text)
        child.chunk_id = _content_id(str(PIPELINE_VERSION), child.doc_id, child.embed_text)


def _make_parent(doc_id, text, start, heading, kind, pages, parent_index, count_tokens, *, derived=False):
    parent = Parent(
        # start keeps two copies of the same paragraph from sharing an id
        parent_id=_content_id(str(PIPELINE_VERSION), doc_id, text, str(start)),
        doc_id=doc_id,
        text=text,
        heading_path=heading,
        block_type=kind,
        start_char=start,
        end_char=start + len(text),
        page_start=min(pages) if pages else 0,
        page_end=max(pages) if pages else 0,
        parent_index=parent_index,
        token_count=count_tokens(text),
        derived=derived or kind == "figure",
    )
    return parent, []


def _section_kind(blocks: list[Block]) -> str:
    kinds = {block.kind for block in blocks}
    if kinds == {"table"}:
        return "table"
    if kinds == {"code"}:
        return "code"
    if kinds == {"row"}:
        return "row"
    if len(kinds) == 1:
        return kinds.pop()
    return "prose"


def _piece_spans(text: str, count_tokens, hard_max: int) -> list[tuple[int, int]]:
    pieces = []
    for start, end in _paragraph_spans(text):
        if count_tokens(text[start:end]) <= hard_max:
            pieces.append((start, end))
        else:
            pieces.extend(_split_long(text, start, end, count_tokens, hard_max))
    return pieces or [(0, len(text))]


def _split_long(text, start, end, count_tokens, hard_max):
    sentences = [span for span in _sentence_spans(text[start:end])]
    units = []
    for local_start, local_end in sentences:
        abs_start = start + local_start
        abs_end = start + local_end
        if count_tokens(text[abs_start:abs_end]) <= hard_max:
            units.append((abs_start, abs_end))
        else:
            # A single sentence longer than the parent cap is cut on words.
            units.extend(_word_spans(text, abs_start, abs_end, count_tokens, hard_max))
    return units


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    while cursor < len(text):
        found = text.find("\n\n", cursor)
        if found < 0:
            spans.append((cursor, len(text)))
            break
        spans.append((cursor, found))
        cursor = found + 2
    return [(start, end) for start, end in spans if text[start:end].strip()]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for match in _SENTENCE.finditer(text):
        end = match.start() + 1 if match.group()[0] in ".!?" else match.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    if not spans and text.strip():
        spans.append((0, len(text)))
    return spans


def _word_spans(text, start, end, count_tokens, hard_max):
    spans = []
    cursor = start
    while cursor < end:
        window_end = cursor
        last = cursor
        while window_end < end:
            nxt = text.find(" ", window_end + 1)
            if nxt < 0 or nxt > end:
                nxt = end
            if count_tokens(text[cursor:nxt]) > hard_max and last > cursor:
                break
            last = nxt
            window_end = nxt
            if nxt == end:
                break
        if last == cursor:
            last = end
        spans.append((cursor, last))
        if last <= cursor:
            break
        cursor = last
        while cursor < end and text[cursor] == " ":
            cursor += 1
    return spans


def _line_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    while cursor <= len(text):
        found = text.find("\n", cursor)
        if found < 0:
            if cursor < len(text):
                spans.append((cursor, len(text)))
            break
        if found > cursor:
            spans.append((cursor, found))
        cursor = found + 1
    return spans or ([(0, len(text))] if text else [])


def _windows(text, spans, count_tokens, target, hard_max, overlap, use_overlap):
    if not spans:
        return []
    if not use_overlap and count_tokens(text[spans[0][0] : spans[-1][1]]) <= hard_max:
        return [(spans[0][0], spans[-1][1])]
    windows = []
    start_i = 0
    while start_i < len(spans):
        end_i = start_i
        tokens = 0
        while end_i < len(spans):
            piece = text[spans[end_i][0] : spans[end_i][1]]
            n = count_tokens(piece)
            if end_i == start_i and n > hard_max:
                end_i = start_i + 1
                break
            if end_i > start_i and (tokens + n > hard_max or tokens >= target):
                break
            tokens += n
            end_i += 1
        if end_i == start_i:
            end_i = start_i + 1
        windows.append((spans[start_i][0], spans[end_i - 1][1]))
        if end_i >= len(spans):
            break
        if use_overlap:
            nxt = _retreat(text, spans, end_i, count_tokens, overlap)
            if nxt <= start_i:
                nxt = start_i + 1
        else:
            nxt = end_i
        start_i = nxt
    return windows


def _retreat(text, spans, end_i, count_tokens, overlap):
    tokens = 0
    index = end_i
    while index > 0:
        n = count_tokens(text[spans[index - 1][0] : spans[index - 1][1]])
        if tokens >= overlap and index < end_i:
            break
        if tokens + n > overlap and tokens > 0:
            break
        index -= 1
        tokens += n
    return index


def _dedupe(children: list[Child]) -> list[Child]:
    seen = set()
    kept = []
    for child in children:
        if child.embed_text in seen:
            continue
        seen.add(child.embed_text)
        kept.append(child)
    return kept


def _content_id(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]
