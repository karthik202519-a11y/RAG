"""FAISS adapter: an exact (flat) index kept in memory and persisted to disk.

Flat means brute-force exact search, so recall is 100% by construction; it is the honest baseline against which
approximate indexes (HNSW, IVF) would be compared. Cosine is implemented as inner product on L2-normalised vectors,
the standard FAISS idiom. Metadata filters are applied *inside* the search through an id selector, so filtered
queries still return k results when k exist.
"""
import json
import threading
from dataclasses import dataclass

import faiss
import numpy as np

from ..config import settings

_lock = threading.RLock()
_cache: dict[str, "_Coll"] = {}


@dataclass
class Hit:
    id: str
    score: float


class _Coll:
    def __init__(self, dim: int, metric: str) -> None:
        self.dim, self.metric = dim, metric
        self.index = faiss.IndexFlatL2(dim) if metric == "euclid" else faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self.meta: list[dict] = []


def _dir():
    d = settings.data_dir / "faiss"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _paths(name: str):
    return _dir() / f"{name}.faiss", _dir() / f"{name}.json"


def mode() -> str:
    return "exact (flat), in-process"


def _load(name: str) -> "_Coll | None":
    with _lock:
        if name in _cache:
            return _cache[name]
        ip, mp = _paths(name)
        if not (ip.exists() and mp.exists()):
            return None
        m = json.loads(mp.read_text(encoding="utf-8"))
        c = _Coll(m["dim"], m["metric"])
        c.index = faiss.read_index(str(ip))
        c.ids, c.meta = m["ids"], m["meta"]
        _cache[name] = c
        return c


def _save(name: str, c: _Coll) -> None:
    ip, mp = _paths(name)
    faiss.write_index(c.index, str(ip))
    mp.write_text(json.dumps({"dim": c.dim, "metric": c.metric, "ids": c.ids, "meta": c.meta}), encoding="utf-8")


def create_collection(name: str, dim: int, metric: str = "cosine") -> None:
    with _lock:
        _cache[name] = _Coll(dim, metric)
        _save(name, _cache[name])


def _prep(c: _Coll, v: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(v, dtype=np.float32)
    if c.metric == "cosine":
        v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    return v


def upsert(name: str, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
    with _lock:
        c = _load(name)
        if c is None:
            raise RuntimeError(f"FAISS collection '{name}' does not exist")
        known = {i: n for n, i in enumerate(c.ids)}
        if any(i in known for i in ids):  # idempotent re-upsert: rebuild without the replaced ids
            keep = [n for n, i in enumerate(c.ids) if i not in set(ids)]
            old = np.vstack([c.index.reconstruct(n) for n in keep]) if keep else np.zeros((0, c.dim), np.float32)
            c.index.reset()
            if len(old):
                c.index.add(old)
            c.ids, c.meta = [c.ids[n] for n in keep], [c.meta[n] for n in keep]
        c.index.add(_prep(c, vectors))
        c.ids += list(ids)
        c.meta += list(payloads)
        _save(name, c)


def search(name: str, vector: np.ndarray, limit: int, document_ids: list[str] | None = None,
           types: list[str] | None = None) -> list[Hit]:
    c = _load(name)
    if c is None or not c.ids:
        return []
    q = _prep(c, np.asarray(vector, dtype=np.float32)[None, :])
    params = None
    if document_ids or types:
        docs, tys = set(document_ids or []), set(types or [])
        pos = [n for n, m in enumerate(c.meta) if (not docs or m.get("document_id") in docs) and (not tys or m.get("ctype") in tys)]
        if not pos:
            return []
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.asarray(pos, dtype=np.int64)))
    scores, idx = c.index.search(q, min(limit, c.index.ntotal), params=params)
    out = []
    for s, n in zip(scores[0], idx[0]):
        if n < 0:
            continue
        # L2 index returns squared distance; report a similarity-like score where larger is better, like Qdrant's euclid
        out.append(Hit(id=c.ids[n], score=float(-s) if c.metric == "euclid" else float(s)))
    return out


def count(name: str) -> int:
    c = _load(name)
    return c.index.ntotal if c else 0


def collection_dim(name: str) -> int | None:
    c = _load(name)
    return c.dim if c else None


def delete_collection(name: str) -> None:
    with _lock:
        _cache.pop(name, None)
        for p in _paths(name):
            p.unlink(missing_ok=True)


def get_vectors(name: str, ids: list[str]) -> dict[str, np.ndarray]:
    """Stored vectors by chunk id. For cosine these are the normalised vectors that the index actually searches."""
    c = _load(name)
    if c is None:
        return {}
    pos = {i: n for n, i in enumerate(c.ids)}
    return {i: np.asarray(c.index.reconstruct(pos[i]), dtype=np.float32) for i in ids if i in pos}


def sample_vectors(name: str, limit: int = 1) -> list[list[float]]:
    c = _load(name)
    return [] if c is None else [list(map(float, c.index.reconstruct(n))) for n in range(min(limit, c.index.ntotal))]
