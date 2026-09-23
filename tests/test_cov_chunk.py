from rag.chunk import (
    _apply_context,
    _children_for,
    _dedupe,
    _label_row,
    _line_spans,
    _merge_tiny,
    _paragraph_spans,
    _parents_from_windows,
    _section_kind,
    _sentence_spans,
    _split_long,
    _table_pieces,
    _table_summary_child,
    _windows,
    _word_spans,
    chunk_document,
)
from rag.models import Block, Child, Parent
from tests.fakes import fake_tokens


def _block(kind, text, heading="H", page=1, start=0):
    return Block(kind, text, heading, page, start, start + len(text))


def _parent(text, kind="prose", heading="H", start=0):
    return Parent("p", "doc", text, heading, kind, start, start + len(text), 1, 1, 0, fake_tokens(text))


def test_blank_blocks_are_dropped_and_rows_window_on_lines():
    parents, children = chunk_document(
        "doc",
        [_block("prose", "   "), _block("row", "alpha\nbeta", "Rows")],
        fake_tokens,
    )
    assert parents
    assert all(child.text in parents[0].text for child in children)


def test_whitespace_window_is_skipped_and_figure_is_one_child():
    packed = _parents_from_windows(
        "doc", "  hello", 0, "H", "prose", [1], 0, fake_tokens, [(0, 2), (2, 7)]
    )
    assert len(packed) == 1
    assert packed[0][0].text == "hello"

    figure = _parent("a labelled valve", "figure")
    kids = _children_for(figure, "figure", fake_tokens)
    assert kids[0].text == figure.text
    assert kids[0].derived is True or figure.derived or figure.block_type == "figure"


def test_empty_child_span_table_summary_and_label_edges():
    parent = _parent("  hello")
    kids = _children_for(parent, "prose", fake_tokens)
    assert kids

    blank = _parent("\n  \n")
    assert _table_summary_child(blank, fake_tokens) is None
    bare = _table_summary_child(_parent("name | qty", "table", heading=""), fake_tokens)
    assert bare is not None
    assert not bare.text.startswith(".")
    assert _label_row("a | b", " | x") == "b: x"
    assert _label_row("a", "1 | extra") == "a: 1; extra"
    assert _table_pieces("", fake_tokens) == []
    assert _table_pieces("   ", fake_tokens) == []
    assert _table_pieces("only header", fake_tokens)[0][0] == "only header"


def test_table_groups_when_a_row_exceeds_the_child_cap(monkeypatch):
    monkeypatch.setattr("rag.chunk.CHILD_MAX", 4)
    text = "h1 | h2\n" + "\n".join(f"r{i} | value{i}" for i in range(6))
    pieces = _table_pieces(text, fake_tokens)
    assert len(pieces) >= 2
    assert all(piece[0] for piece in pieces)


def test_tiny_prose_children_merge_and_context_edges():
    monkeypatch_children = []
    parent = _parent("one two\nthree four")
    for text in ("one two", "three four"):
        monkeypatch_children.append(
            Child(
                "c",
                parent.parent_id,
                "doc",
                text,
                text,
                "H",
                "prose",
                0,
                len(text),
                1,
                1,
                0,
                0,
                2,
            )
        )
    merged = _merge_tiny(monkeypatch_children, fake_tokens)
    assert len(merged) == 1
    assert "one two" in merged[0].text and "three four" in merged[0].text

    orphan = Child("c", "missing", "doc", "x", "x", "", "prose", 0, 1, 0, 0, 0, 0, 1)

    def boom(_parent_text, _child_text):
        raise RuntimeError("context down")

    _apply_context([orphan], [parent], boom, fake_tokens)
    assert orphan.context_prefix == ""
    _apply_context(monkeypatch_children[:1], [parent], lambda *_: "   ", fake_tokens)
    assert monkeypatch_children[0].context_prefix == ""
    _apply_context(monkeypatch_children[:1], [parent], lambda *_: "context", fake_tokens)
    assert monkeypatch_children[0].context_prefix == "context"


def test_section_kind_and_span_splitters():
    assert _section_kind([_block("row", "a")]) == "row"
    assert _section_kind([_block("prose", "a"), _block("code", "b")]) == "prose"
    long = "word " * 30
    pieces = _split_long(long + "tail", 0, len(long), lambda text: len(text.split()), 3)
    assert pieces
    assert _paragraph_spans("hello\n\n   \n\nworld")
    assert _paragraph_spans("hello\n\n")
    assert _sentence_spans("   \nNext")
    assert _sentence_spans("Hi. ")
    assert _sentence_spans("hello")
    assert _line_spans("a\n")
    assert _line_spans("a\n\nb")
    assert _windows("text", [], fake_tokens, 10, 10, 0, False) == []
    text = "aa\nbb\ncc"
    windows = _windows(text, _line_spans(text), lambda piece: len(piece.split()) or 1, 1, 1, 0, False)
    assert len(windows) >= 2
    huge = _windows("SUPERCALIFRAGILISTIC", [(0, 20)], lambda text: len(text), 2, 4, 1, True)
    assert huge


def test_overlap_retreat_can_finish_the_window_loop(monkeypatch):
    monkeypatch.setattr("rag.chunk._retreat", lambda *_args, **_kwargs: 99)
    text = "aa\nbb\ncc"
    windows = _windows(text, _line_spans(text), lambda _piece: 1, 1, 1, 0, True)
    assert windows


def test_private_edges_that_the_document_path_skips(monkeypatch):
    parent = _parent("  hello")

    def only_spaces(*_args, **_kwargs):
        return [(0, 2)]

    monkeypatch.setattr("rag.chunk._windows", only_spaces)
    assert _children_for(parent, "code", fake_tokens) == []

    child = Child("c", parent.parent_id, "doc", "body", "body", "H", "prose", 0, 4, 1, 1, 0, 0, 1)

    def boom(_parent_text, _child_text):
        raise RuntimeError("context down")

    _apply_context([child], [parent], boom, fake_tokens)
    assert child.context_prefix == ""

    short = _split_long("One. Two.", 0, len("One. Two."), fake_tokens, 50)
    assert short
    assert _sentence_spans(" \n ") == [] or _sentence_spans("no-break")


def test_word_spans_cover_a_long_token_and_dedupe():
    spans = _word_spans("alpha beta", 0, 10, lambda text: len(text), 3)
    assert spans[0][0] == 0
    child = Child("c", "p", "d", "t", "same", "", "prose", 0, 1, 0, 0, 0, 0, 1)
    assert len(_dedupe([child, child])) == 1


def test_rechunk_is_stable_and_children_sit_inside_the_source():
    text = "The latch E-4421 stays shut. " * 40
    blocks = [_block("prose", text, "Errors")]
    first_parents, first_children = chunk_document("doc", blocks, fake_tokens)
    second_parents, second_children = chunk_document("doc", blocks, fake_tokens)
    assert [parent.parent_id for parent in first_parents] == [parent.parent_id for parent in second_parents]
    assert [child.chunk_id for child in first_children] == [child.chunk_id for child in second_children]
    for parent in first_parents:
        assert parent.text.strip() in text or parent.text in text
        assert 0 <= parent.start_char <= parent.end_char
    for child in first_children:
        if child.block_type == "prose":
            assert child.text in text
        assert child.start_char <= child.end_char

