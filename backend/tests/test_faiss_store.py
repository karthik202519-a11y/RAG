"""FAISS adapter: unit behaviour plus the same index lifecycle as Qdrant."""
import numpy as np
import pytest

from studio.vectorstores import faiss_store as fs
from test_indexes import build, make_docs, search  # noqa: F401


@pytest.fixture()
def coll(client):  # `client` fixture points data_dir at a temp folder
    name = "unit_coll"
    fs.create_collection(name, 4, "cosine")
    ids = [f"{i:032x}" for i in range(4)]
    vecs = np.array([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    fs.upsert(name, ids, vecs, [{"document_id": "a", "ctype": "text"}, {"document_id": "b", "ctype": "table"},
                                {"document_id": "a", "ctype": "text"}, {"document_id": "b", "ctype": "text"}])
    return name, ids


def test_search_ranking_and_scores(coll):
    name, ids = coll
    hits = fs.search(name, np.array([1, 0, 0, 0], dtype=np.float32), 3)
    assert [h.id for h in hits][:2] == [ids[0], ids[1]]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5) and hits[0].score > hits[1].score


def test_filters_run_inside_the_search(coll):
    name, ids = coll
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    assert {h.id for h in fs.search(name, q, 4, document_ids=["b"])} == {ids[1], ids[3]}
    assert [h.id for h in fs.search(name, q, 4, types=["table"])] == [ids[1]]
    assert fs.search(name, q, 4, document_ids=["zzz"]) == []


def test_upsert_is_idempotent_and_persists(coll):
    name, ids = coll
    fs.upsert(name, [ids[0]], np.array([[0, 0, 0, 1]], dtype=np.float32), [{"document_id": "a", "ctype": "text"}])
    assert fs.count(name) == 4
    fs._cache.clear()  # force a reload from disk
    assert fs.count(name) == 4 and fs.collection_dim(name) == 4
    top = fs.search(name, np.array([0, 0, 0, 1], dtype=np.float32), 1)[0]
    assert top.id == ids[0]
    fs.delete_collection(name)
    assert fs.count(name) == 0 and fs.collection_dim(name) is None


def test_index_on_faiss_matches_qdrant_results(client, auth, project):
    make_docs(client, auth, project)
    a = build(client, auth, project, name="on-qdrant")
    b = build(client, auth, project, name="on-faiss", store="faiss")
    assert b["store"] == "faiss" and b["stored_vectors"] == a["stored_vectors"] > 0
    ra, rb = search(client, auth, a, "Which rocket carried Apollo?", top_k=3), search(client, auth, b, "Which rocket carried Apollo?", top_k=3)
    assert [h["text"] for h in ra["results"]] == [h["text"] for h in rb["results"]]
    first = rb["results"][0]["chunk_id"]
    assert client.get(f"/indexes/{b['id']}/chunks/{first}/vector", headers=auth).status_code == 200
    assert client.delete(f"/indexes/{b['id']}", headers=auth).status_code == 204
    assert fs.count(b["collection"]) == 0
