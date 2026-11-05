"""Real local semantic embeddings and cross-encoder inference against every real store.

Set STUDIO_MODEL_CACHE_DIR to an existing cache to avoid repeated downloads.
"""
import os
import pytest

from conftest import upload, wait_for_job
from studio.config import settings
from studio.vectorstores import STORES
from test_indexes import build


@pytest.mark.parametrize("store", list(STORES))
def test_real_semantic_search_and_rerank(client, auth, project, monkeypatch, store):
    from studio.vectorstores import chroma_store, lancedb_store, pgvector_store
    monkeypatch.setattr(chroma_store, "_client", None)
    monkeypatch.setattr(lancedb_store, "_db", None)
    monkeypatch.setattr(lancedb_store, "_meta", {})
    monkeypatch.setattr(pgvector_store, "_engine", None)
    monkeypatch.setattr(settings, "pgvector_url", os.getenv("TEST_PGVECTOR_URL", "postgresql://postgres:studio@localhost:5434/vec"))
    ok, why = STORES[store].status()
    if not ok:
        pytest.skip(why)
    for name, body in [("france.txt", "Paris is the capital city of France."), ("gardening.txt", "Bananas grow in warm tropical climates.")]:
        doc = upload(client, auth, project["id"], name, body.encode()).json()["results"][0]
        assert wait_for_job(client, auth, doc["job_id"])["status"] == "succeeded"
    idx = build(client, auth, project, provider="fastembed", store=store)
    try:
        assert idx["dimension"] > 0
        r = client.post(f"/indexes/{idx['id']}/retrieve", headers=auth, json={"query": "What is the capital of France?", "rerank": "cross_encoder", "top_k": 2})
        assert r.status_code == 200, r.text
        hits = r.json()["results"]
        assert hits[0]["filename"] == "france.txt"
        assert hits[0]["rerank"]["method"] == "cross_encoder"
        assert hits[0]["rerank"]["score"] > hits[1]["rerank"]["score"]
    finally:
        assert client.delete(f"/indexes/{idx['id']}", headers=auth).status_code == 204
