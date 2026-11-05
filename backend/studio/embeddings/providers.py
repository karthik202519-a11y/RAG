"""Embedding providers. Each returns L2-normalised float32 vectors, so cosine == dot product.

azure_openai  real semantic embeddings (needs AZURE_OPENAI_* settings)
local_hash    offline hashing bag-of-words: deterministic, free, NOT semantic. For demos and tests only.
"""
import hashlib
import re
import time
from dataclasses import dataclass

import numpy as np

from ..config import settings


class ProviderError(RuntimeError):
    """The provider is unavailable or rejected the request."""


class NotConfigured(ProviderError):
    pass


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


@dataclass
class Embedded:
    vectors: np.ndarray
    tokens: int  # tokens the provider billed for this call


class AzureOpenAIEmbedder:
    id = "azure_openai"
    BATCH = 64

    @property
    def model(self) -> str:
        return settings.azure_embedding_deployment

    def status(self) -> tuple[bool, str]:
        missing = [n for n, v in (("AZURE_OPENAI_ENDPOINT", settings.azure_endpoint),
                                  ("AZURE_OPENAI_API_KEY", settings.azure_api_key),
                                  ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", settings.azure_embedding_deployment)) if not v]
        return (not missing, f"Set in .env: {', '.join(missing)}" if missing else "configured")

    def _client(self):
        from openai import OpenAI
        ok, why = self.status()
        if not ok:
            raise NotConfigured(f"Embedding provider 'azure_openai' is not configured. {why}")
        # Foundry endpoints with API version "1" use Azure's v1 route: plain OpenAI client at /openai/v1/.
        return OpenAI(base_url=settings.azure_endpoint.rstrip("/") + "/openai/v1/", api_key=settings.azure_api_key)

    def embed(self, texts: list[str]) -> Embedded:
        client, tokens, out = self._client(), 0, []
        for i in range(0, len(texts), self.BATCH):
            batch = [t if t.strip() else " " for t in texts[i:i + self.BATCH]]
            for attempt in range(5):
                try:
                    r = client.embeddings.create(model=self.model, input=batch)
                    break
                except Exception as e:  # rate limits and transient network errors
                    if attempt == 4:
                        raise ProviderError(f"Azure embedding request failed: {type(e).__name__}") from e
                    time.sleep(2 ** attempt)
            tokens += r.usage.total_tokens
            out.extend(d.embedding for d in r.data)
        return Embedded(_normalize(np.asarray(out, dtype=np.float32)), tokens)


_STOP = set("a an and are as at be by for from how in into is it its of on or that the to was were what when "
            "where which who why with do does".split())


class HashEmbedder:
    id = "local_hash"
    model = "hash-256"
    DIM = 256

    def status(self) -> tuple[bool, str]:
        return True, "always available (offline, not semantic)"

    def embed(self, texts: list[str]) -> Embedded:
        m = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w not in _STOP:
                    m[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        return Embedded(_normalize(m), 0)


PROVIDERS = {p.id: p for p in (AzureOpenAIEmbedder(), HashEmbedder())}
from .providers_extra import EXTRA  # noqa: E402  (imports names defined above)

PROVIDERS.update({p.id: p for p in EXTRA})


def get_provider(provider_id: str):
    p = PROVIDERS.get(provider_id)
    if p is None:
        raise ProviderError(f"Unknown embedding provider '{provider_id}'")
    return p
