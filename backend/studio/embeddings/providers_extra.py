"""More embedding providers. Same contract as providers.py: embed(texts) -> L2-normalised float32 vectors.

fastembed   local semantic embeddings (BGE, E5, MiniLM ... as ONNX): no key, no GPU, downloads the model once
openai      OpenAI API (OPENAI_API_KEY)
ollama      a local Ollama server (OLLAMA_URL)
cohere      Cohere API (COHERE_API_KEY)
voyage      Voyage AI API (VOYAGE_API_KEY)

fastembed is tested against the real model. The hosted providers are tested with recorded responses only: this repository has
no key or server for them, and the Settings page says so.
"""
import threading
import time

import httpx
import numpy as np

from ..config import settings
from .providers import Embedded, NotConfigured, ProviderError, _normalize

BATCH = 64


def _post_json(url: str, headers: dict, body: dict, name: str, timeout: float = 120.0) -> dict:
    for attempt in range(4):
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=timeout)
        except httpx.HTTPError as e:
            if attempt == 3:
                raise ProviderError(f"{name} request failed: {type(e).__name__}") from e
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503) and attempt < 3:
            time.sleep(2 ** attempt)
            continue
        if r.status_code >= 400:
            raise ProviderError(f"{name} returned HTTP {r.status_code}")
        return r.json()
    raise ProviderError(f"{name} request failed")


class FastEmbedEmbedder:
    id = "fastembed"
    _lock = threading.Lock()
    _models: dict = {}

    @property
    def model(self) -> str:
        return settings.fastembed_model

    def status(self) -> tuple[bool, str]:
        try:
            import fastembed  # noqa: F401
        except ImportError:
            return False, "pip install fastembed"
        return True, f"local model {self.model} (downloaded on first use, then offline)"

    def _get(self):
        with self._lock:
            m = self._models.get(self.model)
            if m is None:
                from fastembed import TextEmbedding
                m = self._models[self.model] = TextEmbedding(model_name=self.model, cache_dir=str(settings.data_dir / "models"))
            return m

    def embed(self, texts: list[str]) -> Embedded:
        ok, why = self.status()
        if not ok:
            raise NotConfigured(f"Embedding provider 'fastembed' is not available. {why}")
        try:
            vecs = list(self._get().embed([t if t.strip() else " " for t in texts], batch_size=32))
        except Exception as e:
            raise ProviderError(f"fastembed failed: {type(e).__name__}: {str(e)[:160]}") from e
        return Embedded(_normalize(np.asarray(vecs, dtype=np.float32)), 0)


class OpenAIEmbedder:
    id = "openai"

    @property
    def model(self) -> str:
        return settings.openai_embedding_model

    def status(self) -> tuple[bool, str]:
        return (True, "configured") if settings.openai_api_key else (False, "Set in .env: OPENAI_API_KEY")

    def embed(self, texts: list[str]) -> Embedded:
        ok, why = self.status()
        if not ok:
            raise NotConfigured(f"Embedding provider 'openai' is not configured. {why}")
        out, tokens = [], 0
        for i in range(0, len(texts), BATCH):
            j = _post_json("https://api.openai.com/v1/embeddings", {"Authorization": f"Bearer {settings.openai_api_key}"},
                           {"model": self.model, "input": [t if t.strip() else " " for t in texts[i:i + BATCH]]}, "OpenAI")
            tokens += j.get("usage", {}).get("total_tokens", 0)
            out.extend(d["embedding"] for d in sorted(j["data"], key=lambda d: d["index"]))
        return Embedded(_normalize(np.asarray(out, dtype=np.float32)), tokens)


class OllamaEmbedder:
    id = "ollama"

    @property
    def model(self) -> str:
        return settings.ollama_embedding_model

    def status(self) -> tuple[bool, str]:
        try:
            r = httpx.get(settings.ollama_url.rstrip("/") + "/api/tags", timeout=0.7)
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return False, f"No Ollama server answered at {settings.ollama_url}. Install Ollama and run: ollama pull {self.model}"
        if not any(n.split(":")[0] == self.model.split(":")[0] for n in names):
            return False, f"Ollama is running but model '{self.model}' is not pulled. Run: ollama pull {self.model}"
        return True, f"Ollama at {settings.ollama_url}"

    def embed(self, texts: list[str]) -> Embedded:
        ok, why = self.status()
        if not ok:
            raise NotConfigured(f"Embedding provider 'ollama' is not available. {why}")
        out = []
        for i in range(0, len(texts), BATCH):
            j = _post_json(settings.ollama_url.rstrip("/") + "/api/embed", {}, {"model": self.model, "input": texts[i:i + BATCH]}, "Ollama", 300)
            out.extend(j["embeddings"])
        return Embedded(_normalize(np.asarray(out, dtype=np.float32)), 0)


class CohereEmbedder:
    id = "cohere"

    @property
    def model(self) -> str:
        return settings.cohere_embedding_model

    def status(self) -> tuple[bool, str]:
        return (True, "configured") if settings.cohere_api_key else (False, "Set in .env: COHERE_API_KEY")

    def embed(self, texts: list[str]) -> Embedded:
        ok, why = self.status()
        if not ok:
            raise NotConfigured(f"Embedding provider 'cohere' is not configured. {why}")
        out, tokens = [], 0
        for i in range(0, len(texts), 96):
            j = _post_json("https://api.cohere.com/v2/embed", {"Authorization": f"Bearer {settings.cohere_api_key}"},
                           {"model": self.model, "texts": texts[i:i + 96], "input_type": "search_document", "embedding_types": ["float"]}, "Cohere")
            tokens += int(j.get("meta", {}).get("billed_units", {}).get("input_tokens", 0))
            out.extend(j["embeddings"]["float"])
        return Embedded(_normalize(np.asarray(out, dtype=np.float32)), tokens)


class VoyageEmbedder:
    id = "voyage"

    @property
    def model(self) -> str:
        return settings.voyage_embedding_model

    def status(self) -> tuple[bool, str]:
        return (True, "configured") if settings.voyage_api_key else (False, "Set in .env: VOYAGE_API_KEY")

    def embed(self, texts: list[str]) -> Embedded:
        ok, why = self.status()
        if not ok:
            raise NotConfigured(f"Embedding provider 'voyage' is not configured. {why}")
        out, tokens = [], 0
        for i in range(0, len(texts), 64):
            j = _post_json("https://api.voyageai.com/v1/embeddings", {"Authorization": f"Bearer {settings.voyage_api_key}"},
                           {"model": self.model, "input": texts[i:i + 64], "input_type": "document"}, "Voyage")
            tokens += j.get("usage", {}).get("total_tokens", 0)
            out.extend(d["embedding"] for d in sorted(j["data"], key=lambda d: d["index"]))
        return Embedded(_normalize(np.asarray(out, dtype=np.float32)), tokens)


EXTRA = [FastEmbedEmbedder(), OpenAIEmbedder(), OllamaEmbedder(), CohereEmbedder(), VoyageEmbedder()]
