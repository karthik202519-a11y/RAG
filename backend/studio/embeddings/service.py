"""Embedding with a persistent cache keyed by (provider, model, text)."""
import hashlib
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from concurrent.futures import ThreadPoolExecutor

from ..config import settings
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
    batches = [todo[i:i + step] for i in range(0, len(todo), step)]

    def call(batch):  # network only: sessions are not thread-safe, so no database access in here
        t0 = time.perf_counter()
        return batch, provider.embed([t for _, t in batch]), time.perf_counter() - t0

    # Batches are independent requests: send several at once. Results come back in order and are
    # written to the cache from this thread only.
    workers = max(1, min(settings.embed_concurrency, len(batches)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="embed") as pool:
        for n, (batch, res, secs) in enumerate(pool.map(call, batches), 1):
            spent += secs
            tokens += res.tokens
            for (k, _), vec in zip(batch, res.vectors):
                found[k] = vec
                db.merge(EmbeddingCache(key=k, provider=provider.id, model=provider.model, dim=len(vec),
                                        vector=np.asarray(vec, dtype="<f4").tobytes()))
            db.commit()
            if progress:
                progress(n / len(batches))
    fetched = {k for k, _ in todo}
    hits = sum(1 for k in keys if k not in fetched)
    vectors = np.vstack([found[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)
    return EmbedResult(vectors.astype(np.float32), tokens, hits, spent)
