"""LLM judge with a fake model: the quote check is deterministic and must override a lying judge."""
import json

from studio.generation import judge as J
from studio.generation import service as gen
from test_generation import FakeChat, ask, configured, docs  # noqa: F401
from test_indexes import build

PASSAGES = [{"n": 1, "text": "The Saturn V rocket carried the Apollo spacecraft to the Moon."}]


def _judge_reply(claims, relevance="yes"):
    return json.dumps({"claims": claims, "relevance": relevance})


def test_supported_requires_a_real_quote(configured, monkeypatch):
    reply = _judge_reply([
        {"claim": "Saturn V carried Apollo", "verdict": "supported", "passage": 1, "quote": "Saturn V rocket carried the Apollo"},
        {"claim": "It had five engines", "verdict": "supported", "passage": 1, "quote": "five F-1 engines"},  # invented quote
        {"claim": "It was made of cheese", "verdict": "unsupported", "passage": None, "quote": ""},
    ])
    monkeypatch.setattr(gen, "chat_complete", FakeChat(reply))
    out = J.judge_answer("Which rocket?", "Saturn V [1]. Five engines. Cheese.", PASSAGES)
    assert [c["verdict"] for c in out["claims"]] == ["supported", "unsupported", "unsupported"]
    assert out["claims"][1]["quote_verified"] is False
    assert out["faithfulness"] == round(1 / 3, 3) and out["supported"] == 1 and out["relevance"] == "yes"


def test_malformed_json_is_an_error(configured, monkeypatch):
    monkeypatch.setattr(gen, "chat_complete", FakeChat("not json at all"))
    try:
        J.judge_answer("q", "a", PASSAGES)
        assert False, "expected GenerationError"
    except gen.GenerationError:
        pass


def test_judge_endpoint(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    reply = _judge_reply([{"claim": "Saturn V carried Apollo", "verdict": "supported", "passage": 1, "quote": "carried the Apollo spacecraft"}])
    monkeypatch.setattr(gen, "chat_complete", FakeChat(reply))
    r = client.post(f"/indexes/{idx['id']}/judge", headers=auth, json={"question": "q", "answer": "Saturn V [1]", "passages": PASSAGES})
    assert r.status_code == 200, r.text
    assert r.json()["faithfulness"] == 1.0
    bad = client.post(f"/indexes/{idx['id']}/judge", headers=auth, json={"question": "q", "answer": "a", "passages": [{"text": "x"}]})
    assert bad.status_code == 422
