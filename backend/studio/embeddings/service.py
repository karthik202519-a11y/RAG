"""Embedding with a persistent cache keyed by (provider, model, text)."""
import hashlib
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import EmbeddingCache
from .providers import get_provider

_IN_LIMIT = 500  # keep IN (...) lists small for SQLite


def cache_key(provider: str, model: str, text: str) -> str:
    return hashlib.sha256(f"{provider}\x1f{model}\x1f{text}".encode()).hexdigest()


@dataclass
class EmbedResult:
    vectors: np.ndarray
    tokens_billed: int
    cache_hits: int
    seconds_in_provider: float = 0.0


def embed_texts(db: Session, provider_id: str, texts: list[str], progress=None) -> EmbedResult:
    """Return one vector per text. Cached texts cost nothing; the rest are sent in provider batches."""
    import time
    provider = get_provider(provider_id)
    keys = [cache_key(provider.id, provider.model, t) for t in texts]
    found: dict[str, np.ndarray] = {}
    uniq = list(dict.fromkeys(keys))
    for i in range(0, len(uniq), _IN_LIMIT):
        for row in db.scalars(select(EmbeddingCache).where(EmbeddingCache.key.in_(uniq[i:i + _IN_LIMIT]))):
            found[row.key] = np.frombuffer(row.vector, dtype="<f4")
    todo = [(k, t) for k, t in dict.fromkeys(zip(keys, texts)) if k not in found]
    tokens, spent = 0, 0.0
    step = 64
    for i in range(0, len(todo), step):
        chunk = todo[i:i + step]
        t0 = time.perf_counter()
        res = provider.embed([t for _, t in chunk])
        spent += time.perf_counter() - t0
        tokens += res.tokens
        for (k, _), vec in zip(chunk, res.vectors):
            found[k] = vec
            db.merge(EmbeddingCache(key=k, provider=provider.id, model=provider.model, dim=len(vec),
                                    vector=np.asarray(vec, dtype="<f4").tobytes()))
        db.commit()
        if progress:
            progress(min(1.0, (i + step) / max(1, len(todo))))
    fetched = {k for k, _ in todo}
    hits = sum(1 for k in keys if k not in fetched)
    vectors = np.vstack([found[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)
    return EmbedResult(vectors.astype(np.float32), tokens, hits, spent)
