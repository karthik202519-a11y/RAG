"""Embeddings: Azure OpenAI / AI Foundry (production) or an offline hashing embedder (tests).

A hash-keyed cache (data/embedding_cache.jsonl) means a given text is embedded, and
billed, only once, across documents, chunk sizes and re-runs.
"""
import hashlib
import json
import re
import threading
import time

import numpy as np

from . import config

BATCH = 64
FAKE_DIM = 256
_STOPWORDS = set("a an and are as at be by for from how in into is it its of on or that the to was were "
                 "what when where which who why with do does".split())


class EmbeddingError(RuntimeError):
    """Embeddings are unavailable (not configured, or the service failed)."""


def embed_text(chunk: dict) -> str:
    """What the model sees. Text chunks get their section title as context;
    table chunks already start with it."""
    if chunk["type"] == "text" and chunk.get("section"):
        return f"Section: {chunk['section']}\n{chunk['text']}"
    return chunk["text"]


def key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---------------------------------------------------------------- fake backend
def _fake_embed(texts: list[str]) -> np.ndarray:
    """Bag-of-words hashing: deterministic, offline, good enough to test plumbing."""
    out = np.zeros((len(texts), FAKE_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            if w in _STOPWORDS:
                continue
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % FAKE_DIM] += 1.0
    return _normalize(out)


# --------------------------------------------------------------- azure backend
class _AzureCache:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.vectors: dict[str, np.ndarray] | None = None

    @property
    def path(self):
        return config.DATA_DIR / "embedding_cache.jsonl"

    def load(self) -> dict[str, np.ndarray]:
        if self.vectors is None:
            self.vectors = {}
            if self.path.exists():
                with self.path.open(encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        self.vectors[r["k"]] = np.asarray(r["v"], dtype=np.float32)
        return self.vectors


_cache = _AzureCache()


def _client():
    from openai import OpenAI

    missing = [n for n, v in (("AZURE_OPENAI_ENDPOINT", config.AZURE_ENDPOINT),
                              ("AZURE_OPENAI_API_KEY", config.AZURE_API_KEY),
                              ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", config.AZURE_EMBEDDING_DEPLOYMENT)) if not v]
    if missing:
        raise EmbeddingError(f"Embeddings not configured; set in .env: {', '.join(missing)}")
    # Foundry endpoints with API version "1" use Azure's v1 route: the plain OpenAI
    # client at <endpoint>/openai/v1/, deployment name passed as `model`.
    return OpenAI(base_url=config.AZURE_ENDPOINT.rstrip("/") + "/openai/v1/", api_key=config.AZURE_API_KEY)


def _azure_embed(texts: list[str]) -> tuple[np.ndarray, int]:
    """Return (normalized matrix, tokens billed this call)."""
    with _cache.lock:
        cache = _cache.load()
        todo = list(dict.fromkeys(t for t in texts if key(t) not in cache))
        tokens = 0
        if todo:
            client = _client()
            _cache.path.parent.mkdir(parents=True, exist_ok=True)
            with _cache.path.open("a", encoding="utf-8") as f:
                for i in range(0, len(todo), BATCH):
                    batch = todo[i:i + BATCH]
                    for attempt in range(5):
                        try:
                            r = client.embeddings.create(model=config.AZURE_EMBEDDING_DEPLOYMENT, input=batch)
                            break
                        except Exception as e:  # rate limit / transient
                            if attempt == 4:
                                raise EmbeddingError(f"Embedding request failed: {type(e).__name__}") from e
                            time.sleep(2 ** attempt)
                    tokens += r.usage.total_tokens
                    for t, d in zip(batch, r.data):
                        cache[key(t)] = np.asarray(d.embedding, dtype=np.float32)
                        f.write(json.dumps({"k": key(t), "v": d.embedding}) + "\n")
        mat = np.vstack([cache[key(t)] for t in texts])
    return _normalize(mat), tokens


# -------------------------------------------------------------------- public API
def embed_texts(texts: list[str]) -> tuple[np.ndarray, int]:
    """Embed texts; returns (L2-normalized float32 matrix, tokens billed)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), 0
    if config.EMBEDDING_BACKEND == "fake":
        return _fake_embed(texts), 0
    if config.EMBEDDING_BACKEND == "azure":
        return _azure_embed(texts)
    raise EmbeddingError(f"Unknown RAG_EMBEDDING_BACKEND '{config.EMBEDDING_BACKEND}' (use azure or fake)")
