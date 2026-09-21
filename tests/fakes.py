import hashlib
import math


def fake_tokens(text: str) -> int:
    words = text.split()
    return len(words) if words else 0


def fake_encode(texts, *, query=False):
    vectors = []
    for text in texts:
        prefix = "q:" if query else "d:"
        digest = hashlib.sha256((prefix + text).encode()).digest()
        raw = [(digest[index % len(digest)] / 127.5) - 1 for index in range(8)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        vectors.append([value / norm for value in raw])
    return vectors
