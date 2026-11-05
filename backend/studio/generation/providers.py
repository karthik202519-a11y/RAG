"""Chat providers besides Azure OpenAI. A model name with a provider prefix routes here: `openai:gpt-4o-mini`, `anthropic:claude-haiku-4-5`,
`gemini:gemini-2.0-flash`, `ollama:llama3.2`. A name without a prefix is an Azure deployment, as before.

These adapters are tested against recorded responses in the documented shape of each API, not against the live services (this
repository has no keys for them). Bedrock is not offered: it needs AWS request signing and credentials.
"""
import time

import httpx

from ..config import settings

PREFIXES = ("openai", "anthropic", "gemini", "ollama")


class ProviderChatError(RuntimeError):
    pass


def split_model(model: str) -> tuple[str | None, str]:
    head, sep, tail = model.partition(":")
    return (head, tail) if sep and head in PREFIXES else (None, model)


def status(provider: str) -> tuple[bool, str]:
    if provider == "openai":
        return (True, "configured") if settings.openai_api_key else (False, "Set in .env: OPENAI_API_KEY")
    if provider == "anthropic":
        return (True, "configured") if settings.anthropic_api_key else (False, "Set in .env: ANTHROPIC_API_KEY")
    if provider == "gemini":
        return (True, "configured") if settings.gemini_api_key else (False, "Set in .env: GEMINI_API_KEY")
    if provider == "ollama":
        try:
            httpx.get(settings.ollama_url.rstrip("/") + "/api/tags", timeout=0.7).raise_for_status()
            return True, f"Ollama at {settings.ollama_url}"
        except Exception:
            return False, f"No Ollama server answered at {settings.ollama_url}"
    return False, f"Unknown provider '{provider}'"


def default_models() -> list[str]:
    """Prefixed model names that are usable right now, for the model picker."""
    pairs = (("openai", settings.openai_chat_model), ("anthropic", settings.anthropic_chat_model),
             ("gemini", settings.gemini_chat_model), ("ollama", settings.ollama_chat_model))
    return [f"{p}:{m}" for p, m in pairs if m and status(p)[0]]


def _post(url: str, headers: dict, body: dict, name: str) -> dict:
    for attempt in range(3):
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=180)
        except httpx.HTTPError as e:
            if attempt == 2:
                raise ProviderChatError(f"{name} request failed ({type(e).__name__}).") from e
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503) and attempt < 2:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            raise ProviderChatError(f"{name}: model not found.")
        if r.status_code >= 400:
            raise ProviderChatError(f"{name} request failed (HTTP {r.status_code}).")
        return r.json()
    raise ProviderChatError(f"{name} request failed.")


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


def chat(provider: str, model: str, messages: list[dict], temperature: float, max_output_tokens: int):
    """-> (text, prompt_tokens, completion_tokens, finish_reason, served_model)"""
    ok, why = status(provider)
    if not ok:
        raise ProviderChatError(f"{provider} is not configured. {why}")
    if provider == "openai":
        body = {"model": model, "messages": messages, "max_completion_tokens": max_output_tokens}
        if not model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
            body["temperature"] = temperature
        j = _post("https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {settings.openai_api_key}"}, body, "OpenAI")
        ch = j["choices"][0]
        return ch["message"].get("content") or "", j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"], ch.get("finish_reason"), j.get("model", model)
    if provider == "anthropic":
        system, rest = _split_system(messages)
        body = {"model": model, "max_tokens": max_output_tokens, "temperature": temperature, "messages": rest}
        if system:
            body["system"] = system
        j = _post("https://api.anthropic.com/v1/messages", {"x-api-key": settings.anthropic_api_key, "anthropic-version": "2023-06-01"}, body, "Anthropic")
        text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        stop = {"end_turn": "stop", "max_tokens": "length"}.get(j.get("stop_reason"), j.get("stop_reason"))
        return text, j["usage"]["input_tokens"], j["usage"]["output_tokens"], stop, j.get("model", model)
    if provider == "gemini":
        system, rest = _split_system(messages)
        body = {"contents": [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in rest],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        j = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", {"x-goog-api-key": settings.gemini_api_key}, body, "Gemini")
        cand = j["candidates"][0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
        u = j.get("usageMetadata", {})
        fin = {"STOP": "stop", "MAX_TOKENS": "length"}.get(cand.get("finishReason"), cand.get("finishReason"))
        return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0), fin, j.get("modelVersion", model)
    j = _post(settings.ollama_url.rstrip("/") + "/api/chat", {}, {"model": model, "messages": messages, "stream": False,
                                                                 "options": {"temperature": temperature, "num_predict": max_output_tokens}}, "Ollama")
    return j["message"]["content"], j.get("prompt_eval_count", 0), j.get("eval_count", 0), {"stop": "stop", "length": "length"}.get(j.get("done_reason"), j.get("done_reason")), model
