"""Answer generation with a fake chat model (tests never call Azure)."""
import pytest

from conftest import register
from studio.config import settings
from studio.generation import REFUSAL, check_citations
from studio.generation import service as gen
from test_indexes import CHUNKING, build, make_docs  # noqa: F401  (fixtures and helpers)


@pytest.fixture()
def docs(client, auth, project):
    return make_docs(client, auth, project)


@pytest.fixture()
def configured(monkeypatch):
    monkeypatch.setattr(settings, "azure_endpoint", "https://example.invalid")
    monkeypatch.setattr(settings, "azure_api_key", "not-a-real-key")
    monkeypatch.setattr(settings, "azure_chat_deployment", "fake-chat")
    monkeypatch.setattr(settings, "azure_chat_deployments", "fake-chat,fake-big")


class FakeChat:
    def __init__(self, reply="The Saturn V carried Apollo [1]."):
        self.reply, self.calls = reply, []

    def __call__(self, model, messages, temperature, max_output_tokens):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature, "max": max_output_tokens})
        return gen.ChatResult(self.reply, prompt_tokens=321, completion_tokens=12, reasoning_tokens=0,
                              finish_reason="stop", model=f"{model}-served")


def ask(client, auth, idx, **body):
    return client.post(f"/indexes/{idx['id']}/answer", headers=auth, json={"question": "Which rocket carried Apollo?", **body})


def test_not_configured_is_a_clear_409(client, auth, project, docs):
    idx = build(client, auth, project)
    r = ask(client, auth, idx)
    assert r.status_code == 409 and "not configured" in r.json()["detail"]
    g = client.get("/providers", headers=auth).json()["generation"]
    assert g["configured"] is False and "AZURE_OPENAI" in g["detail"]


def test_grounded_answer_with_inspectable_prompt_and_resolved_citations(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    fake = FakeChat()
    monkeypatch.setattr(gen, "chat_complete", fake)
    r = ask(client, auth, idx, top_k=3).json()
    assert r["answer"] == "The Saturn V carried Apollo [1]." and r["abstained"] is False
    assert r["citations"] == {"cited": [1], "unresolved": [], "all_resolve": True, "has_citations": True}
    sys_msg, user_msg = r["prompt"]["messages"]
    assert sys_msg["role"] == "system" and REFUSAL in sys_msg["content"] and "untrusted data" in sys_msg["content"]
    assert user_msg["content"].count("[1] ") >= 1 and "Question: Which rocket carried Apollo?" in user_msg["content"]
    passages = r["context_passages"]
    assert passages[0]["n"] == 1 and passages[0]["filename"] == "apollo.txt" and "Saturn V" in passages[0]["text"]
    assert r["usage"]["prompt_tokens"] == 321 and r["usage"]["served_by"] == "fake-chat-served"
    assert fake.calls[0]["model"] == "fake-chat" and fake.calls[0]["temperature"] == 0.0
    assert r["timings_ms"]["retrieval_ms"] >= 0 and r["timings_ms"]["llm_ms"] >= 0


def test_unresolved_citation_markers_are_flagged(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    monkeypatch.setattr(gen, "chat_complete", FakeChat("It was the Saturn V [1][9]."))
    r = ask(client, auth, idx, top_k=3).json()
    assert r["citations"]["all_resolve"] is False and r["citations"]["unresolved"] == [9]


def test_model_refusal_is_reported_as_abstention(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    monkeypatch.setattr(gen, "chat_complete", FakeChat(REFUSAL))
    r = ask(client, auth, idx).json()
    assert r["abstained"] is True and r["citations"]["has_citations"] is False


def test_min_score_abstains_without_calling_the_model(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    fake = FakeChat()
    monkeypatch.setattr(gen, "chat_complete", fake)
    r = ask(client, auth, idx, question="zebra quantum pastry", min_score=0.99).json() if False else \
        client.post(f"/indexes/{idx['id']}/answer", headers=auth, json={"question": "zebra quantum pastry", "min_score": 0.99}).json()
    assert r["abstained"] is True and "below the minimum" in r["abstain_reason"] and r["usage"] is None
    assert fake.calls == []  # no tokens spent


def test_context_budget_drops_lower_ranked_passages_and_says_so(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    monkeypatch.setattr(gen, "chat_complete", FakeChat())
    r = ask(client, auth, idx, top_k=10, context_budget=60).json()
    assert len(r["context_passages"]) >= 1 and r["dropped_passages"]
    assert r["prompt"]["context_tokens"] <= 60 + 40  # the first passage is always kept, even if it alone is over budget
    assert [p["n"] for p in r["context_passages"]] == list(range(1, len(r["context_passages"]) + 1))


def test_open_mode_uses_a_different_system_prompt(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    monkeypatch.setattr(gen, "chat_complete", FakeChat())
    r = ask(client, auth, idx, strict=False).json()
    assert REFUSAL not in r["prompt"]["messages"][0]["content"] and r["strict"] is False


def test_request_validation_and_model_override(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    fake = FakeChat()
    monkeypatch.setattr(gen, "chat_complete", fake)
    assert ask(client, auth, idx, top_k=0).status_code == 422
    assert ask(client, auth, idx, temperature=5).status_code == 422
    assert ask(client, auth, idx, max_output_tokens=1).status_code == 422
    assert client.post(f"/indexes/{idx['id']}/answer", headers=auth, json={"question": ""}).status_code == 422
    assert ask(client, auth, idx, model="fake-big").status_code == 200 and fake.calls[-1]["model"] == "fake-big"


def test_provider_failures_are_502_and_leak_nothing(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)

    def boom(*a, **k):
        raise gen.GenerationError("Chat request failed (APIConnectionError).")

    monkeypatch.setattr(gen, "chat_complete", boom)
    r = ask(client, auth, idx)
    assert r.status_code == 502 and "not-a-real-key" not in r.text


def test_generation_respects_access_control(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    monkeypatch.setattr(gen, "chat_complete", FakeChat())
    eve = register(client, "eve@example.com", "Eve")
    assert ask(client, eve, idx).status_code == 404
    assert client.post(f"/indexes/{idx['id']}/answer", json={"question": "x"}).status_code == 401


def test_providers_report_generation_models(client, auth, configured):
    g = client.get("/providers", headers=auth).json()["generation"]
    assert g["configured"] is True and g["default_model"] == "fake-chat" and g["models"] == ["fake-chat", "fake-big"]


def test_citation_checker_edge_cases():
    assert check_citations("no markers here", 3) == {"cited": [], "unresolved": [], "all_resolve": True, "has_citations": False}
    assert check_citations("a [2] b [2] c [0]", 3)["unresolved"] == [0]
    assert check_citations("see [1] and [10]", 3)["unresolved"] == [10]
    assert check_citations("array[abc] and [x]", 3)["has_citations"] is False


def test_reasoning_models_are_detected_so_temperature_is_not_sent():
    assert gen.is_reasoning_model("gpt-5-mini") and gen.is_reasoning_model("o3-mini")
    assert not gen.is_reasoning_model("gpt-4o-mini") and not gen.is_reasoning_model("gpt-4.1")
