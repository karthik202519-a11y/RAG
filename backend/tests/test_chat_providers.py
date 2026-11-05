"""Chat provider adapters, tested against recorded responses in each API's documented shape (no live keys here)."""
import httpx
import pytest

from studio.config import settings
from studio.generation import providers
from studio.generation import service as gen


class Resp:
    def __init__(self, body, code=200):
        self.body, self.status_code = body, code

    def json(self):
        return self.body


@pytest.fixture()
def keys(monkeypatch):
    for k in ("openai_api_key", "anthropic_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, k, "test-key")


def capture(monkeypatch, body, code=200):
    calls = []

    def fake(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json})
        return Resp(body, code)
    monkeypatch.setattr(httpx, "post", fake)
    return calls


MSGS = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


def test_prefix_routing():
    assert providers.split_model("anthropic:claude-x") == ("anthropic", "claude-x")
    assert providers.split_model("gpt-4o-mini") == (None, "gpt-4o-mini")
    assert providers.split_model("unknown:thing") == (None, "unknown:thing")


def test_anthropic_request_and_response(keys, monkeypatch):
    calls = capture(monkeypatch, {"model": "claude-x", "content": [{"type": "text", "text": "Hello"}], "stop_reason": "end_turn",
                                  "usage": {"input_tokens": 11, "output_tokens": 3}})
    r = gen.chat_complete("anthropic:claude-x", MSGS, 0.2, 50)
    assert (r.text, r.prompt_tokens, r.completion_tokens, r.finish_reason) == ("Hello", 11, 3, "stop")
    c = calls[0]
    assert c["url"].endswith("/v1/messages") and c["headers"]["x-api-key"] == "test-key" and c["headers"]["anthropic-version"]
    assert c["json"]["system"] == "Be brief." and c["json"]["messages"] == [{"role": "user", "content": "Hi"}] and c["json"]["max_tokens"] == 50


def test_gemini_request_and_response(keys, monkeypatch):
    calls = capture(monkeypatch, {"candidates": [{"content": {"parts": [{"text": "Hey"}]}, "finishReason": "MAX_TOKENS"}],
                                  "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2}, "modelVersion": "gemini-x-001"})
    r = gen.chat_complete("gemini:gemini-x", MSGS, 0.0, 20)
    assert (r.text, r.prompt_tokens, r.completion_tokens, r.finish_reason, r.model) == ("Hey", 7, 2, "length", "gemini-x-001")
    assert ":generateContent" in calls[0]["url"] and calls[0]["json"]["systemInstruction"]["parts"][0]["text"] == "Be brief."
    assert calls[0]["json"]["generationConfig"]["maxOutputTokens"] == 20


def test_openai_request_and_response(keys, monkeypatch):
    calls = capture(monkeypatch, {"model": "gpt-4o-mini-2024", "choices": [{"message": {"content": "Yo"}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    r = gen.chat_complete("openai:gpt-4o-mini", MSGS, 0.0, 30)
    assert r.text == "Yo" and calls[0]["json"]["temperature"] == 0.0 and calls[0]["headers"]["Authorization"] == "Bearer test-key"
    gen.chat_complete("openai:gpt-5-mini", MSGS, 0.0, 30)
    assert "temperature" not in calls[1]["json"]  # reasoning models accept only the default


def test_ollama_request_and_response(monkeypatch):
    monkeypatch.setattr(providers, "status", lambda p: (True, "ok"))
    calls = capture(monkeypatch, {"message": {"content": "Local"}, "prompt_eval_count": 9, "eval_count": 4, "done_reason": "stop"})
    r = gen.chat_complete("ollama:llama3.2", MSGS, 0.0, 40)
    assert (r.text, r.prompt_tokens, r.completion_tokens) == ("Local", 9, 4)
    assert calls[0]["url"].endswith("/api/chat") and calls[0]["json"]["options"]["num_predict"] == 40 and calls[0]["json"]["stream"] is False


def test_missing_key_and_http_errors_are_clear(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    with pytest.raises(gen.GenerationError, match="ANTHROPIC_API_KEY"):
        gen.chat_complete("anthropic:claude-x", MSGS, 0.0, 10)
    monkeypatch.setattr(settings, "anthropic_api_key", "k")
    capture(monkeypatch, {"error": "bad"}, 401)
    with pytest.raises(gen.GenerationError, match="HTTP 401"):
        gen.chat_complete("anthropic:claude-x", MSGS, 0.0, 10)
    ok, why = gen.status_for("gemini:gemini-x")
    assert not ok and "GEMINI_API_KEY" in why
