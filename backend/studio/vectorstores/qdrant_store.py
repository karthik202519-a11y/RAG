"""Qdrant adapter. Server mode when QDRANT_URL is set, otherwise an embedded on-disk instance
(the real Qdrant engine, in-process, single process)."""
import threading
import uuid
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models

from ..config import settings

METRICS = {"cosine": models.Distance.COSINE, "dot": models.Distance.DOT, "euclid": models.Distance.EUCLID}
_lock = threading.Lock()
_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    with _lock:
        if _client is None:
            if settings.qdrant_url:
                _client = QdrantClient(url=settings.qdrant_url, timeout=30)
            else:
                path = settings.qdrant_path or str(settings.data_dir / "qdrant")
                _client = QdrantClient(path=path)
        return _client


def reset_client() -> None:
    global _client
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
        _client = None


@dataclass
class Hit:
    id: str
    score: float


def mode() -> str:
    return "server" if settings.qdrant_url else "embedded"


def create_collection(name: str, dim: int, metric: str = "cosine") -> None:
    c = client()
    if c.collection_exists(name):
        c.delete_collection(name)
    c.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=METRICS[metric]))


def upsert(name: str, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
    pts = [models.PointStruct(id=str(uuid.UUID(i)), vector=v.tolist(), payload=p)
           for i, v, p in zip(ids, vectors, payloads)]
    client().upsert(name, points=pts, wait=True)


def search(name: str, vector: np.ndarray, limit: int, document_ids: list[str] | None = None,
           types: list[str] | None = None) -> list[Hit]:
    must = []
    if document_ids:
        must.append(models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids)))
    if types:
        must.append(models.FieldCondition(key="ctype", match=models.MatchAny(any=types)))
    res = client().query_points(name, query=vector.tolist(), limit=limit, with_payload=False,
                                query_filter=models.Filter(must=must) if must else None)
    return [Hit(id=uuid.UUID(str(p.id)).hex, score=float(p.score)) for p in res.points]


def count(name: str) -> int:
    return client().count(name, exact=True).count


def collection_dim(name: str) -> int | None:
    try:
        cfg = client().get_collection(name).config.params.vectors
        return int(cfg.size)  # type: ignore[union-attr]
    except Exception:
        return None


def delete_collection(name: str) -> None:
    c = client()
    if c.collection_exists(name):
        c.delete_collection(name)


def sample_vectors(name: str, limit: int = 1) -> list[list[float]]:
    pts, _ = client().scroll(name, limit=limit, with_vectors=True, with_payload=False)
    return [list(map(float, p.vector)) for p in pts]  # type: ignore[arg-type]


def get_vectors(name: str, ids: list[str]) -> dict[str, np.ndarray]:
    """Stored vectors by chunk id (hex)."""
    pts = client().retrieve(name, ids=[str(uuid.UUID(i)) for i in ids], with_vectors=True)
    return {uuid.UUID(str(p.id)).hex: np.asarray(p.vector, dtype=np.float32) for p in pts}
