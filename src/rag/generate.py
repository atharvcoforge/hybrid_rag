import json
import os
import urllib.error
import urllib.request

SYSTEM = (
    "Answer using only the passages. "
    "Each passage begins with its number in brackets. Cite only those numbers. "
    "If the passages do not contain the answer, reply exactly: The documents do not say."
)


def generator_url() -> str:
    return os.environ.get("GENERATOR_URL", "http://host.docker.internal:8081/v1").rstrip("/")


def writer_up() -> bool:
    base = generator_url()
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as resp:
            return resp.status == 200
    except (OSError, urllib.error.URLError):
        return False


def pack(question: str, hits) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        pages = f"pp. {hit.page_start}-{hit.page_end}" if hit.page_start else ""
        blocks.append(f"[{number}] {hit.source_path} {hit.heading_path} {pages}\n{hit.parent_text}")
    return "\n\n".join(blocks) + "\n\nQuestion: " + question


def _body(question: str, hits, stream: bool) -> dict:
    return {
        "model": os.environ.get("GENERATOR_MODEL", "qwen2.5-3b-instruct"),
        "temperature": 0,
        "max_tokens": 380,
        "stream": stream,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": pack(question, hits)},
        ],
    }


def _request(question: str, hits, stream: bool):
    payload = json.dumps(_body(question, hits, stream)).encode()
    return urllib.request.Request(
        generator_url() + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )


def stream_answer(question: str, hits):
    with urllib.request.urlopen(_request(question, hits, True), timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                yield delta


def complete(question: str, hits) -> str:
    with urllib.request.urlopen(_request(question, hits, False), timeout=180) as resp:
        payload = json.loads(resp.read().decode())
    return payload["choices"][0]["message"]["content"]
