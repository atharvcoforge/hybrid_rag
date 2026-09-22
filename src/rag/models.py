from dataclasses import dataclass

PIPELINE_VERSION = 2

PARENT_TARGET = 700
PARENT_MAX = 900
PARENT_OVERLAP = 120
CHILD_TARGET = 180
CHILD_MAX = 280
CHILD_OVERLAP = 40

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_CHUNKS = 8000

RRF_K = 60
DENSE_K = 20
BM25_K = 20
RERANK_K = 12
AGREE_TOP = 3
FAST_GAP = 0.60
SCORE_BAND = 0.15
MAX_PARENTS = 5
TAU_KEEP = 0.95

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBED_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
EMBED_BATCH = 32
EMBED_DIM = 1024


class IngestError(Exception):
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}" if path else reason)


class QueryError(Exception):
    pass


@dataclass
class Block:
    kind: str
    text: str
    heading_path: str
    page: int
    start_char: int
    end_char: int
    derived: bool = False
    ocr: bool = False
    ocr_confidence: float | None = None
    flagged: bool = False


@dataclass
class Parent:
    parent_id: str
    doc_id: str
    text: str
    heading_path: str
    block_type: str
    start_char: int
    end_char: int
    page_start: int
    page_end: int
    parent_index: int
    token_count: int
    derived: bool = False


@dataclass
class Child:
    chunk_id: str
    parent_id: str
    doc_id: str
    text: str
    embed_text: str
    heading_path: str
    block_type: str
    start_char: int
    end_char: int
    page_start: int
    page_end: int
    child_index: int
    parent_index: int
    token_count: int
    derived: bool = False
    context_prefix: str = ""


@dataclass
class Hit:
    parent_id: str
    parent_text: str
    heading_path: str
    source_path: str
    file_sha256: str
    page_start: int
    page_end: int
    start_char: int
    end_char: int
    child_id: str
    score: float
    confident: bool
    derived: bool = False
    ocr: bool = False


@dataclass
class Retrieval:
    hits: list[Hit]
    reason: str = ""
    stages_ms: dict | None = None
    warnings: list[str] | None = None


@dataclass
class Ingested:
    doc_id: str
    status: str
    chunks: int
    warnings: list[str] | None = None


def tau_key(embed_model: str, embed_revision: str, mode: str = "cascade") -> str:
    return "|".join((embed_model, embed_revision, RERANK_MODEL, RERANK_REVISION, mode))


def make_embed_text(heading_path: str, body: str) -> str:
    if heading_path:
        return heading_path + "\n" + body
    return body
