from rag.models import Retrieval
from rag.server import iter_query


def test_abstain_does_not_call_the_writer():
    def search(_text, _mode):
        return Retrieval(hits=[], reason="no_confident_hit")

    def write(_text, _hits):
        raise AssertionError("writer called")

    events = list(iter_query("what is the salary", search, write, "cascade"))
    assert events[0][0] == "meta"
    assert events[0][1]["reason"] == "no_confident_hit"
    assert events[-1] == ("done", {"first_token_ms": None, "answer": ""})
