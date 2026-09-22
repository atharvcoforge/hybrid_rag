"""Chunker depth: table labels, summary child, tiny-child merge (§04)."""

from rag.chunk import chunk_document
from rag.models import Block
from rag.parse import assign_offsets
from tests.fakes import fake_tokens


def _blocks(*rows):
    blocks = [Block(kind, text, heading, page, 0, 0) for kind, text, heading, page in rows]
    return assign_offsets(blocks)


def test_table_rows_carry_column_labels():
    blocks = _blocks(("table", "Region | tCO2e\nIndia | 14,644", "Emissions", 0))
    _parents, children = chunk_document("doc", blocks, fake_tokens)
    row_kids = [c for c in children if "14,644" in c.text]
    assert row_kids
    assert any("Region: India" in c.text and "tCO2e: 14,644" in c.text for c in row_kids)


def test_table_gets_a_summary_child():
    blocks = _blocks(("table", "Region | tCO2e\nIndia | 14,644\nUK | 8,688", "Emissions", 0))
    _parents, children = chunk_document("doc", blocks, fake_tokens)
    summaries = [c for c in children if c.block_type == "table_summary"]
    assert len(summaries) == 1
    assert "Region" in summaries[0].text
    assert "tCO2e" in summaries[0].text


def test_tiny_children_merge_unless_they_hold_an_identifier():
    blocks = _blocks(
        ("prose", "Hi.\nE-4421 means open.\nBye now friend.", "Errors", 0),
    )
    _parents, children = chunk_document("doc", blocks, fake_tokens)
    # "Hi." alone is too small and has no identifier → merged.
    assert not any(c.text.strip() == "Hi." for c in children)
    # Error code must stay findable on its own span.
    assert any("E-4421" in c.text for c in children)


def test_contextual_prefix_is_prepended_to_embed_text_only():
    blocks = _blocks(("prose", "Total: 14,644", "India FY24", 0))

    def context_fn(parent_text, child_text):
        del parent_text, child_text
        return "India FY24 baseline emissions in tCO2e."

    _parents, children = chunk_document(
        "doc", blocks, fake_tokens, context_fn=context_fn
    )
    assert children[0].text == "Total: 14,644"
    assert children[0].context_prefix.startswith("India FY24")
    assert children[0].embed_text.startswith("India FY24")
    assert "Total: 14,644" in children[0].embed_text
