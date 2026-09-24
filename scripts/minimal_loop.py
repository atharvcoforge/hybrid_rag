"""Embed two known texts, store them, and retrieve the more relevant one.

This is the early embed-store-retrieve check. It uses the same encoder and
SQLite index as the full pipeline, on a throwaway directory.
"""

import tempfile

from rag.embed import encode_documents, encode_query
from rag.models import EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION, Child, Parent
from rag.store import Index

LEFT = "The housing clicks into the rail until the tab seats."
RIGHT = "SKU-7842-XL ships in a plain carton with no tools."
QUERY = "How does the housing attach to the rail?"


def _child(number: int, text: str) -> tuple[Child, Parent]:
    parent = Parent(
        parent_id=f"p{number}",
        doc_id="notes.txt",
        text=text,
        heading_path="Note",
        block_type="prose",
        start_char=0,
        end_char=len(text),
        page_start=1,
        page_end=1,
        parent_index=number,
        token_count=len(text.split()),
    )
    child = Child(
        chunk_id=f"c{number}",
        parent_id=parent.parent_id,
        doc_id="notes.txt",
        text=text,
        embed_text=text,
        heading_path="Note",
        block_type="prose",
        start_char=0,
        end_char=len(text),
        page_start=1,
        page_end=1,
        child_index=0,
        parent_index=number,
        token_count=len(text.split()),
    )
    return child, parent


def main() -> None:
    directory = tempfile.mkdtemp(prefix="rag-two-text-")
    index = Index(directory, EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION)
    index.open()
    try:
        pairs = [_child(0, LEFT), _child(1, RIGHT)]
        children = [child for child, _parent in pairs]
        parents = [parent for _child, parent in pairs]
        vectors = encode_documents([child.embed_text for child in children])
        index.upsert(
            children,
            parents,
            vectors,
            {
                "source_path": "notes.txt",
                "filename": "notes.txt",
                "mime": "text/plain",
                "file_sha256": "two-text",
                "pipeline_version": PIPELINE_VERSION,
            },
        )
        hits = index.dense_search(encode_query(QUERY), 2)
    finally:
        index.close()
    if len(hits) != 2:
        raise SystemExit(f"expected 2 hits, got {len(hits)}")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}  {hit['dense_score']:.4f}  {hit['embed_text']}")
    if LEFT not in hits[0]["embed_text"]:
        raise SystemExit("the rail sentence was not the top hit")
    print("top hit is the rail sentence")


if __name__ == "__main__":
    main()
