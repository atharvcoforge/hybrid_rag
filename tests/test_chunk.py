import re

from rag.chunk import chunk_document
from rag.models import Block
from rag.parse import assign_offsets
from tests.fakes import fake_tokens


def _blocks(*rows):
    blocks = [Block(kind, text, heading, page, 0, 0) for kind, text, heading, page in rows]
    return assign_offsets(blocks)


def test_short_section_is_one_parent_and_keeps_the_heading_on_the_embedding():
    blocks = _blocks(("prose", "E-4421 means the latch is open.", "Errors", 0))
    parents, children = chunk_document("doc", blocks, fake_tokens)
    assert len(parents) == 1
    assert len(children) == 1
    assert children[0].text == "E-4421 means the latch is open."
    assert children[0].embed_text == "Errors\n" + children[0].text
    assert children[0].parent_id == parents[0].parent_id
    assert children[0].start_char == parents[0].start_char


def test_ids_stay_put_when_the_text_does_not_change():
    blocks = _blocks(("prose", "The valve stays shut.", "Methods", 0))
    first, _ = chunk_document("doc", blocks, fake_tokens)
    second, _ = chunk_document("doc", blocks, fake_tokens)
    assert [parent.parent_id for parent in first] == [parent.parent_id for parent in second]


def test_table_is_not_folded_into_the_prose_around_it():
    blocks = _blocks(
        ("prose", "Intro sentence.", "H", 0),
        ("table", "sku | name\nSKU-1 | alpha", "H", 0),
        ("prose", "After the table.", "H", 0),
    )
    parents, _children = chunk_document("doc", blocks, fake_tokens)
    table = next(parent for parent in parents if parent.block_type == "table")
    assert "Intro" not in table.text
    assert "After" not in table.text
    assert "SKU-1" in table.text


def test_table_rows_stay_whole_and_keep_the_header(monkeypatch):
    monkeypatch.setattr("rag.chunk.CHILD_MAX", 8)
    monkeypatch.setattr("rag.chunk.CHILD_TARGET", 6)
    blocks = _blocks(("table", "sku | name\nSKU-1 | alpha one\nSKU-2 | gamma two", "Parts", 0))
    _parents, children = chunk_document("doc", blocks, fake_tokens)
    row_kids = [c for c in children if c.block_type == "table"]
    assert len(row_kids) == 2
    assert all(child.text.startswith("sku | name") for child in row_kids)
    assert any("SKU-1" in child.text and "SKU-2" not in child.text for child in row_kids)
    assert any("SKU-2" in child.text and "SKU-1" not in child.text for child in row_kids)


def test_code_splits_on_lines(monkeypatch):
    monkeypatch.setattr("rag.chunk.CHILD_MAX", 3)
    monkeypatch.setattr("rag.chunk.CHILD_TARGET", 3)
    blocks = _blocks(("code", "alpha line stays\nbeta line stays", "Setup", 0))
    _parents, children = chunk_document("doc", blocks, fake_tokens)
    assert {child.text for child in children} == {"alpha line stays", "beta line stays"}


def test_duplicate_embed_text_is_kept_once():
    blocks = _blocks(
        ("code", "export PATH=/opt/bin", "Setup", 0),
        ("code", "export PATH=/opt/bin", "Setup", 0),
    )
    _parents, children = chunk_document("doc", blocks, fake_tokens)
    assert len(children) == 1


def test_long_section_overlaps_on_paragraphs():
    paragraphs = [f"para{i} " + " ".join(["word"] * 40) for i in range(30)]
    blocks = _blocks(("prose", "\n\n".join(paragraphs), "Chapter", 0))
    parents, _children = chunk_document("doc", blocks, fake_tokens)
    assert len(parents) >= 2
    assert "para0" in parents[0].text
    shared = set(parents[0].text.split()) & set(parents[1].text.split())
    assert any(word.startswith("para") for word in shared)


def test_sentences_are_not_cut_in_half():
    text = " ".join(f"Sentence{i} stays whole today." for i in range(80))
    blocks = _blocks(("prose", text, "Notes", 0))
    _parents, children = chunk_document("doc", blocks, fake_tokens)
    assert len(children) >= 2
    for child in children:
        for name in re.findall(r"Sentence\d+", child.text):
            assert f"{name} stays whole today." in child.text


def test_a_sentence_longer_than_the_parent_cap_is_cut_on_words(monkeypatch):
    monkeypatch.setattr("rag.chunk.PARENT_MAX", 10)
    monkeypatch.setattr("rag.chunk.PARENT_TARGET", 8)
    monkeypatch.setattr("rag.chunk.PARENT_OVERLAP", 2)
    words = [f"w{i}" for i in range(30)]
    blocks = _blocks(("prose", " ".join(words), "Notes", 0))
    parents, _children = chunk_document("doc", blocks, fake_tokens)
    assert len(parents) >= 2
    assert all(parent.token_count <= 10 for parent in parents)
    covered = " ".join(parent.text for parent in parents)
    assert all(word in covered.split() for word in words)


def test_energy_and_waste_sections_are_different_parents():
    from rag.parse import pdf_prose_blocks

    text = (
        "Energy Optimization and Emission Management\n"
        'We, at Coforge, are committed to become "Carbon Neutral in our operations by 2040".\n'
        "Procure 10% of electricity from green sources by 2025.\n"
        "Waste Management and Circularity\n"
        "Coforge has committed to become Zero Waste by 2040."
    )
    blocks = assign_offsets(pdf_prose_blocks(text, 4))
    parents, _children = chunk_document("envi", blocks, fake_tokens)
    energy = [parent for parent in parents if "Carbon Neutral in our operations by 2040" in parent.text]
    assert len(energy) == 1
    assert "Zero Waste" not in energy[0].text
    assert energy[0].heading_path == "Energy Optimization and Emission Management"
    waste = [parent for parent in parents if "Zero Waste" in parent.text]
    assert len(waste) == 1
    assert "Carbon Neutral" not in waste[0].text


def test_a_name_and_role_stay_one_parent():
    from rag.parse import pdf_prose_blocks

    text = "John Speight\nPresident and Head of Europe (EVP)"
    blocks = assign_offsets(pdf_prose_blocks(text, 3))
    parents, _children = chunk_document("carbon", blocks, fake_tokens)
    assert len(parents) == 1
    assert "John Speight" in parents[0].text
    assert "President and Head of Europe" in parents[0].text
