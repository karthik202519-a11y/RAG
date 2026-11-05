"""Chroma adapter (embedded, on disk). HNSW index; metadata filters run inside the query."""
import threading
from dataclasses import dataclass

import numpy as np

from ..config import settings

_lock = threading.RLock()
_client = None
SPACES = {"cosine": "cosine", "dot": "ip", "euclid": "l2"}


@dataclass
class Hit:
    id: str
    score: float


def status() -> tuple[bool, str]:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False, "pip install chromadb"
    return True, "embedded, on disk"


def mode() -> str:
    return "embedded (HNSW), on disk"


def client():
    global _client
    with _lock:
        if _client is None:
            import chromadb
            path = settings.data_dir / "chroma"
            path.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(path=str(path))
        return _client


def reset_client() -> None:
    global _client
    _client = None


def _coll(name: str):
    return client().get_collection(name)


def create_collection(name: str, dim: int, metric: str = "cosine") -> None:
    c = client()
    try:
        c.delete_collection(name)
    except Exception:
        pass
    c.create_collection(name, metadata={"hnsw:space": SPACES[metric], "dim": dim})


def upsert(name: str, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
    col = _coll(name)
    meta = [{"document_id": p.get("document_id", ""), "ctype": p.get("ctype", ""), "chunk_index": int(p.get("chunk_index", 0))} for p in payloads]
    for i in range(0, len(ids), 4000):  # Chroma caps the size of one write
        col.upsert(ids=ids[i:i + 4000], embeddings=np.asarray(vectors[i:i + 4000], dtype=np.float32).tolist(), metadatas=meta[i:i + 4000])


def search(name: str, vector: np.ndarray, limit: int, document_ids: list[str] | None = None, types: list[str] | None = None) -> list[Hit]:
    col = _coll(name)
    n = col.count()
    if n == 0:
        return []
    conds = []
    if document_ids:
        conds.append({"document_id": {"$in": list(document_ids)}})
    if types:
        conds.append({"ctype": {"$in": list(types)}})
    where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}
    res = col.query(query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()], n_results=min(limit, n), where=where)
    space = (col.metadata or {}).get("hnsw:space", "cosine")
    out = []
    for i, d in zip(res["ids"][0], res["distances"][0]):
        # Chroma returns distances (smaller is closer); report similarity so larger is better, as the other stores do.
        out.append(Hit(id=i, score=float(-d) if space == "l2" else float(1.0 - d)))
    return out


def count(name: str) -> int:
    try:
        return _coll(name).count()
    except Exception:
        return 0


def collection_dim(name: str) -> int | None:
    try:
        return int((_coll(name).metadata or {}).get("dim")) or None
    except Exception:
        return None


def delete_collection(name: str) -> None:
    try:
        client().delete_collection(name)
    except Exception:
        pass


def get_vectors(name: str, ids: list[str]) -> dict[str, np.ndarray]:
    res = _coll(name).get(ids=list(ids), include=["embeddings"])
    return {i: np.asarray(v, dtype=np.float32) for i, v in zip(res["ids"], res["embeddings"])}


def sample_vectors(name: str, limit: int = 1) -> list[list[float]]:
    res = _coll(name).get(limit=limit, include=["embeddings"])
    return [list(map(float, v)) for v in res["embeddings"]]
