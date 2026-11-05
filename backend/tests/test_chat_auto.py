"""Automatic key selection stays deterministic and never calls a service to select it."""
import pytest

from studio.config import settings
from studio.generation import providers, service as gen


@pytest.fixture(autouse=True)
def isolate_selection(monkeypatch):
    for field in ("gemini_api_key", "nvidia_api_key", "openai_api_key", "anthropic_api_key",
                  "azure_endpoint", "azure_api_key", "azure_chat_deployment", "azure_chat_deployments"):
        monkeypatch.setattr(settings, field, "")
    monkeypatch.setattr(settings, "disable_azure", False)
    monkeypatch.setattr(settings, "chat_model", "auto")
    monkeypatch.setattr(settings, "vision_model", "")
    monkeypatch.setattr(providers, "default_models", lambda: [])


@pytest.mark.parametrize("fields,expected", [
    ({}, ""),
    ({"gemini_api_key": "google"}, "gemini:"),
    ({"nvidia_api_key": "nvidia"}, "nvidia:"),
    ({"openai_api_key": "openai"}, "openai:"),
    ({"gemini_api_key": "google", "nvidia_api_key": "nvidia", "openai_api_key": "openai"}, "gemini:"),
    ({"nvidia_api_key": "nvidia", "openai_api_key": "openai"}, "nvidia:"),
    ({"azure_endpoint": "https://example.invalid", "azure_api_key": "azure", "azure_chat_deployment": "existing-deployment"}, "existing-deployment"),
])
def test_auto_selects_available_key(monkeypatch, fields, expected):
    for field, value in fields.items():
        monkeypatch.setattr(settings, field, value)
    chosen = gen.default_model()
    assert chosen.startswith(expected) if expected.endswith(":") else chosen == expected


def test_explicit_model_is_not_replaced(monkeypatch):
    monkeypatch.setattr(settings, "chat_model", "openai:chosen-model")
    monkeypatch.setattr(settings, "gemini_api_key", "google")
    assert gen.default_model() == "openai:chosen-model"


def test_disabled_azure_remains_disabled_with_existing_key(monkeypatch):
    monkeypatch.setattr(settings, "disable_azure", True)
    monkeypatch.setattr(settings, "azure_endpoint", "https://example.invalid")
    monkeypatch.setattr(settings, "azure_api_key", "azure")
    monkeypatch.setattr(settings, "azure_chat_deployment", "existing")
    assert gen.default_model() == ""
    from studio.embeddings.providers import get_provider
    assert not get_provider("azure_openai").status()[0]


def test_nvidia_transport_and_usage(monkeypatch):
    monkeypatch.setattr(settings, "nvidia_api_key", "secret-not-output")
    calls = []

    def post(url, headers, body, name):
        calls.append((url, headers, body))
        return {"choices": [{"message": {"content": "Evidence [1]."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4}, "model": "catalog-model"}

    monkeypatch.setattr(providers, "_post", post)
    result = gen.chat_complete("nvidia:catalog-model", [{"role": "user", "content": "question"}], 0, 100)
    assert result.text == "Evidence [1]." and result.prompt_tokens == 12
    assert calls[0][0] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert calls[0][2]["max_tokens"] == 100
    assert "max_completion_tokens" not in calls[0][2]


def test_google_vision_uses_inline_png_and_excludes_thought_parts(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "secret-not-output")
    monkeypatch.setattr(settings, "vision_model", "gemini:gemma-4-26b-a4b-it")
    calls = []

    def post(url, headers, body, name):
        calls.append(body)
        return {"candidates": [{"content": {"parts": [{"text": "private thought", "thought": True}, {"text": "red"}]}, "finishReason": "STOP"}]}

    monkeypatch.setattr(providers, "_post", post)
    result = gen.chat_vision(gen.default_vision_model(), "Describe", b"png", 80)
    assert result.text == "red"
    assert calls[0]["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/png"
    assert calls[0]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"


def test_settings_never_exposes_added_key(client, auth, monkeypatch):
    monkeypatch.setattr(settings, "nvidia_api_key", "NVIDIA-SECRET-DO-NOT-RETURN")
    r = client.get("/settings/overview", headers=auth)
    assert r.status_code == 200 and "NVIDIA-SECRET-DO-NOT-RETURN" not in r.text
    assert next(p for p in r.json()["hosted_setup"] if p["id"] == "nvidia")["key_set"]
