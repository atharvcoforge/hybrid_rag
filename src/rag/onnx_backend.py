"""Optional ONNX Runtime backends for embed/rerank (§08 latency).

Sentence-transformers remains the default. Set RAG_ONNX_EMBED / RAG_ONNX_RERANK
to an .onnx path to prefer the quantized graph when present.
"""

from __future__ import annotations

import os
from pathlib import Path


def onnx_embed_path() -> Path | None:
    raw = os.environ.get("RAG_ONNX_EMBED", "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def onnx_rerank_path() -> Path | None:
    raw = os.environ.get("RAG_ONNX_RERANK", "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def try_onnx_session(path: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
