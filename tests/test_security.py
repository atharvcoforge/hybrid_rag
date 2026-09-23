import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from rag.server import (
    _MAX_QUERY_BYTES,
    IngestBody,
    enforce_query_limit,
    eval_post,
    require_token,
)


def test_token_is_required_even_when_unset(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    with pytest.raises(HTTPException) as missing:
        require_token(None)
    assert missing.value.status_code == 401
    monkeypatch.setenv("API_TOKEN", "secret")
    with pytest.raises(HTTPException) as wrong:
        require_token("Bearer nope")
    assert wrong.value.status_code == 401
    require_token("Bearer secret")


def test_ingest_body_rejects_a_path_field():
    with pytest.raises(ValidationError):
        IngestBody.model_validate({"path": "/etc"})
    with pytest.raises(ValidationError):
        IngestBody.model_validate({"path": "../.."})
    assert IngestBody.model_validate({}).model_dump() == {}


def test_megabyte_query_is_413():
    enforce_query_limit("Who signed the plan?")
    with pytest.raises(HTTPException) as exc:
        enforce_query_limit("x" * (_MAX_QUERY_BYTES + 1))
    assert exc.value.status_code == 413


def test_eval_route_returns_429_when_the_queue_is_full():
    from rag.server import _queue

    held = 0
    try:
        while held < _queue.maxsize:
            _queue.acquire()
            held += 1
        with pytest.raises(HTTPException) as exc:
            eval_post()
        assert exc.value.status_code == 429
    finally:
        for _ in range(held):
            _queue.release()
