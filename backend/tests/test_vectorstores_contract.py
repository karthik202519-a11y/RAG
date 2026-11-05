"""Every vector store must behave the same: ranking, filters, idempotent upsert, vector retrieval, delete.
pgvector runs only when PGVECTOR_URL points at a database that has the extension (the Docker image pgvector/pgvector:pg16)."""
import os
import uuid

import numpy as np
import pytest

from studio.config import settings
from studio.vectorstores import STORES

TEST_PG = os.getenv("TEST_PGVECTOR_URL", "postgresql://postgres:studio@localhost:5434/vec")


@pytest.fixture(params=["qdrant", "faiss", "chroma", "lancedb", "pgvector"])
def store(request, client, monkeypatch):
    name = request.param
    if name == "pgvector":
        monkeypatch.setattr(settings, "pgvector_url", TEST_PG)
        from studio.vectorstores import pgvector_store
        pgvector_store._engine = None
        ok, why = pgvector_store.status()
        if not ok:
            pytest.skip(f"pgvector unavailable: {why}")
    return STORES[name]


IDS = [uuid.UUID(int=i + 1).hex for i in range(5)]
VECS = np.array([[1, 0, 0, 0], [0.9, 0.2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.7, 0.7, 0, 0]], dtype=np.float32)
META = [{"document_id": "a", "ctype": "text"}, {"document_id": "b", "ctype": "table"}, {"document_id": "a", "ctype": "text"},
        {"document_id": "b", "ctype": "text"}, {"document_id": "a", "ctype": "table"}]


def fill(store, name):
    store.delete_collection(name)
    store.create_collection(name, 4, "cosine")
    store.upsert(name, IDS, VECS, META)


def test_ranking_scores_count_and_dim(store):
    name = f"contract_{uuid.uuid4().hex[:8]}"
    fill(store, name)
    try:
        hits = store.search(name, np.array([1, 0, 0, 0], dtype=np.float32), 3)
        assert [h.id for h in hits] == [IDS[0], IDS[1], IDS[4]]
        assert hits[0].score == pytest.approx(1.0, abs=1e-3) and hits[0].score > hits[1].score > hits[2].score
        assert store.count(name) == 5 and store.collection_dim(name) == 4
        assert len(store.search(name, np.array([1, 0, 0, 0], dtype=np.float32), 50)) == 5
    finally:
        store.delete_collection(name)
    assert store.count(name) == 0


def test_filters_by_document_and_type(store):
    name = f"contract_{uuid.uuid4().hex[:8]}"
    fill(store, name)
    try:
        q = np.array([1, 0, 0, 0], dtype=np.float32)
        assert {h.id for h in store.search(name, q, 5, document_ids=["b"])} == {IDS[1], IDS[3]}
        assert [h.id for h in store.search(name, q, 5, types=["table"], document_ids=["a"])] == [IDS[4]]
        assert store.search(name, q, 5, document_ids=["nope"]) == []
    finally:
        store.delete_collection(name)


def test_upsert_is_idempotent_and_vectors_come_back(store):
    name = f"contract_{uuid.uuid4().hex[:8]}"
    fill(store, name)
    try:
        store.upsert(name, [IDS[0]], np.array([[0, 0, 0, 1]], dtype=np.float32), [META[0]])
        assert store.count(name) == 5
        top = store.search(name, np.array([0, 0, 0, 1], dtype=np.float32), 1)[0]
        assert top.id == IDS[0]
        got = store.get_vectors(name, [IDS[2], IDS[3]])
        assert set(got) == {IDS[2], IDS[3]}
        v = got[IDS[2]]
        assert v.shape == (4,) and abs(float(np.linalg.norm(v)) - 1.0) < 1e-3 and v.argmax() == 1
        assert len(store.sample_vectors(name, 2)) == 2
    finally:
        store.delete_collection(name)


def test_status_and_mode_are_reported(store):
    ok, why = store.status()
    assert ok and why and store.mode()
