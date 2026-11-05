"""Indexes, embeddings, vector store and retrieval (offline hash embedder + embedded Qdrant)."""
import numpy as np
import pytest

from conftest import make_pdf, register, upload, wait_for_job
from studio.vectorstores import qdrant_store

CHUNKING = {"method": "recursive", "params": {"size": 25, "overlap": 4}}


def make_docs(client, auth, project):
    texts = {
        "apollo.txt": "The Saturn V rocket carried the Apollo spacecraft toward lunar orbit.\n\nApollo 11 landed first, on 20 July 1969, "
                      "in the Sea of Tranquility. Neil Armstrong and Buzz Aldrin walked on the Moon.",
        "plants.txt": "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen inside chloroplasts.\n\n"
                      "Chlorophyll absorbs red and blue light and reflects green light.",
        "database.txt": "A relational database stores rows in tables and uses SQL to query them.\n\n"
                        "An index speeds up lookups at the cost of extra storage. Transaction ZX-4471 groups several operations.",
    }
    ids = {}
    for name, body in texts.items():
        r = upload(client, auth, project["id"], name, body.encode()).json()["results"][0]
        assert wait_for_job(client, auth, r["job_id"])["status"] == "succeeded"
        ids[name] = r["document_id"]
    return ids


def build(client, auth, project, name="idx", chunking=CHUNKING, provider="local_hash", **extra):
    r = client.post(f"/projects/{project['id']}/indexes", headers=auth,
                    json={"name": name, "chunking": chunking, "provider": provider, **extra})
    assert r.status_code == 201, r.text
    idx = r.json()
    job = wait_for_job(client, auth, idx["job"]["id"])
    assert job["status"] == "succeeded", job
    return client.get(f"/indexes/{idx['id']}", headers=auth).json()


def search(client, auth, idx, query, **kw):
    r = client.post(f"/indexes/{idx['id']}/search", headers=auth, json={"query": query, **kw})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture()
def docs(client, auth, project):
    return make_docs(client, auth, project)


def test_providers_report_availability_without_secrets(client, auth):
    p = client.get("/providers", headers=auth).json()
    by = {e["id"]: e for e in p["embedding"]}
    assert by["local_hash"]["available"] is True and by["local_hash"]["semantic"] is False
    assert by["azure_openai"]["available"] is False and "AZURE_OPENAI" in by["azure_openai"]["detail"]
    assert "key" not in str(p).lower().replace("api_key", "")  # no secret values anywhere
    assert p["vector_stores"][0]["id"] == "qdrant"


def test_build_index_stores_vectors_and_chunks(client, auth, project, docs):
    idx = build(client, auth, project)
    assert idx["status"] == "ready" and idx["dimension"] == 256 and idx["metric"] == "cosine"
    assert idx["chunk_count"] > 3 and idx["stored_vectors"] == idx["chunk_count"]
    assert set(idx["chunks_by_document"]) == {"apollo.txt", "plants.txt", "database.txt"}
    assert idx["collection"] == f"idx_{idx['id']}" and idx["documents"] == 3


def test_dense_search_returns_right_document_with_citation(client, auth, project, docs):
    idx = build(client, auth, project)
    r = search(client, auth, idx, "Which rocket carried Apollo toward the Moon?", top_k=3)
    top = r["results"][0]
    assert top["filename"] == "apollo.txt" and "Saturn V" in top["text"]
    assert top["citation"].startswith("apollo.txt") and top["rank"] == 1
    assert top["contributions"]["dense"]["rank"] == 1 and 0 < top["score"] <= 1
    qe = r["info"]["query_embedding"]
    assert qe["dimension"] == 256 and abs(qe["norm"] - 1.0) < 1e-4 and qe["provider"] == "local_hash"
    assert r["timings_ms"]["vector_search_ms"] >= 0 and r["run_id"]


def test_bm25_finds_exact_identifier_and_reports_matched_terms(client, auth, project, docs):
    idx = build(client, auth, project)
    r = search(client, auth, idx, "ZX-4471", mode="bm25", top_k=3)
    top = r["results"][0]
    assert top["filename"] == "database.txt" and "ZX-4471" in top["text"]
    assert top["contributions"]["bm25"]["matched_terms"] == ["zx-4471"]
    assert "dense" not in top["contributions"] and "query_embedding" not in r["info"]


def test_hybrid_fuses_both_rankings(client, auth, project, docs):
    idx = build(client, auth, project)
    r = search(client, auth, idx, "Sea of Tranquility landing Apollo", mode="hybrid", top_k=4)
    top = r["results"][0]
    assert top["filename"] == "apollo.txt"
    c = top["contributions"]
    assert {"dense", "bm25", "rrf"} <= set(c)
    assert c["rrf"] == pytest.approx(1 / (60 + c["dense"]["rank"]) + 1 / (60 + c["bm25"]["rank"]), abs=1e-6)
    scores = [x["score"] for x in r["results"]]
    assert scores == sorted(scores, reverse=True)


def test_filters_and_threshold(client, auth, project, docs):
    idx = build(client, auth, project)
    only = search(client, auth, idx, "light", mode="dense", top_k=10, document_ids=[docs["plants.txt"]])
    assert only["results"] and all(x["document_id"] == docs["plants.txt"] for x in only["results"])
    tables = search(client, auth, idx, "database", top_k=10, types=["table"])
    assert tables["results"] == []  # no tables in these documents
    hard = search(client, auth, idx, "Saturn rocket", top_k=10, score_threshold=0.999)
    assert hard["results"] == []
    assert client.post(f"/indexes/{idx['id']}/search", headers=auth, json={"query": "x", "mode": "bogus"}).status_code == 422


def test_indexes_are_isolated_versions_and_reuse_the_embedding_cache(client, auth, project, docs):
    a = build(client, auth, project, "A")
    b = build(client, auth, project, "B", chunking={"method": "fixed_token", "params": {"size": 40, "overlap": 5}})
    c = build(client, auth, project, "C")  # identical config to A
    assert len({a["collection"], b["collection"], c["collection"]}) == 3
    assert a["chunk_count"] != b["chunk_count"]
    assert c["cache_hits"] == c["chunk_count"]  # every text was already embedded for A
    assert client.delete(f"/indexes/{a['id']}", headers=auth).status_code == 204
    assert qdrant_store.count(b["collection"]) == b["chunk_count"]  # B untouched
    assert client.get(f"/indexes/{a['id']}", headers=auth).status_code == 404
    assert search(client, auth, b, "Apollo")["results"]
    assert not qdrant_store.client().collection_exists(a["collection"])


def test_query_dimension_must_match_the_index(client, auth, project, docs, monkeypatch):
    idx = build(client, auth, project)
    from studio.embeddings.providers import HashEmbedder
    monkeypatch.setattr(HashEmbedder, "DIM", 128)  # the provider now produces 128-d vectors
    r = client.post(f"/indexes/{idx['id']}/search", headers=auth, json={"query": "Apollo"})
    assert r.status_code == 409 and "128" in r.json()["detail"] and "256" in r.json()["detail"]
    assert client.post(f"/indexes/{idx['id']}/search", headers=auth, json={"query": "Apollo", "mode": "bm25"}).status_code == 200


def test_unconfigured_provider_is_refused_clearly(client, auth, project, docs):
    r = client.post(f"/projects/{project['id']}/indexes", headers=auth,
                    json={"name": "x", "chunking": CHUNKING, "provider": "azure_openai"})
    assert r.status_code == 409 and "not configured" in r.json()["detail"]
    r = client.post(f"/projects/{project['id']}/indexes", headers=auth,
                    json={"name": "x", "chunking": CHUNKING, "provider": "nope"})
    assert r.status_code == 422
    assert client.post("/providers/azure_openai/test", headers=auth).status_code == 409


def test_create_validation(client, auth, project):
    body = {"name": "x", "chunking": CHUNKING, "provider": "local_hash"}
    r = client.post(f"/projects/{project['id']}/indexes", headers=auth, json=body)
    assert r.status_code == 422 and "No ready documents" in r.json()["detail"]
    make_docs(client, auth, project)
    for chunking, msg in [({"method": "fixed_token", "params": {"size": 50, "overlap": 50}}, "Overlap"),
                          ({"method": "topic_based", "params": {}}, "not available"),
                          ({"method": "nope", "params": {}}, "Unknown")]:
        r = client.post(f"/projects/{project['id']}/indexes", headers=auth, json={**body, "chunking": chunking})
        assert r.status_code == 422 and msg in r.json()["detail"], r.text


def test_index_access_control(client, auth, project, docs):
    idx = build(client, auth, project)
    eve = register(client, "eve@example.com", "Eve")
    assert client.get(f"/indexes/{idx['id']}", headers=eve).status_code == 404
    assert client.post(f"/indexes/{idx['id']}/search", headers=eve, json={"query": "Apollo"}).status_code == 404
    assert client.delete(f"/indexes/{idx['id']}", headers=eve).status_code == 404
    assert client.get(f"/projects/{project['id']}/indexes", headers=eve).status_code == 404
    assert client.get(f"/indexes/{idx['id']}/chunks", headers=eve).status_code == 404
    assert client.post(f"/indexes/{idx['id']}/search", json={"query": "x"}).status_code == 401


def test_failed_build_is_reported_and_cannot_be_searched(client, auth, project, docs, monkeypatch):
    from studio.embeddings import NotConfigured
    from studio.jobs import index_job

    def boom(*a, **k):
        raise NotConfigured("provider went away")

    monkeypatch.setattr(index_job, "embed_texts", boom)
    r = client.post(f"/projects/{project['id']}/indexes", headers=auth,
                    json={"name": "doomed", "chunking": CHUNKING, "provider": "local_hash"}).json()
    job = wait_for_job(client, auth, r["job"]["id"])
    assert job["status"] == "dead" and "provider went away" in job["error"]
    idx = client.get(f"/indexes/{r['id']}", headers=auth).json()
    assert idx["status"] == "failed" and "provider went away" in idx["error"]
    s = client.post(f"/indexes/{idx['id']}/search", headers=auth, json={"query": "Apollo"})
    assert s.status_code == 409 and "failed" in s.json()["detail"]


def test_browse_chunks_vector_and_runs(client, auth, project, docs):
    idx = build(client, auth, project)
    page = client.get(f"/indexes/{idx['id']}/chunks", headers=auth, params={"limit": 2}).json()
    assert page["total"] == idx["chunk_count"] and len(page["items"]) == 2
    v = client.get(f"/indexes/{idx['id']}/chunks/{page['items'][0]['id']}/vector", headers=auth).json()
    assert v["dimension"] == 256 and v["norm"] == pytest.approx(1.0, abs=1e-4) and len(v["first_values"]) == 16
    assert client.get(f"/indexes/{idx['id']}/chunks/{'0' * 32}/vector", headers=auth).status_code == 404
    search(client, auth, idx, "Apollo")
    search(client, auth, idx, "database", mode="bm25")
    runs = client.get(f"/indexes/{idx['id']}/runs", headers=auth).json()
    assert [r["mode"] for r in runs] == ["bm25", "dense"] and runs[0]["results"] > 0


def test_pdf_chunks_carry_pages_for_citations(client, auth, project):
    pdf = make_pdf([["First page about volcanoes and magma rising through the crust."],
                    ["Second page about glaciers and slow ice flow across valleys."]])
    r = upload(client, auth, project["id"], "geo.pdf", pdf).json()["results"][0]
    assert wait_for_job(client, auth, r["job_id"])["status"] == "succeeded"
    idx = build(client, auth, project, chunking={"method": "page_aware", "params": {"max_tokens": 0}})
    top = search(client, auth, idx, "glaciers ice", mode="bm25")["results"][0]
    assert top["pages"] == [2] and "PDF p.2" in top["citation"]


def test_embedding_vectors_are_unit_length_and_deterministic(client, auth):
    from studio.embeddings.providers import HashEmbedder
    a = HashEmbedder().embed(["the quick brown fox", "lazy dogs sleep"]).vectors
    b = HashEmbedder().embed(["the quick brown fox", "lazy dogs sleep"]).vectors
    assert np.allclose(a, b) and np.allclose(np.linalg.norm(a, axis=1), 1.0)


# ------------------------------------------------------------- parent-child and neighbour expansion
PC = {"method": "parent_child", "params": {"parent_size": 90, "child_size": 25, "child_overlap": 4}}


def test_parent_child_index_stores_parents_and_search_can_return_them(client, auth, project, docs):
    idx = build(client, auth, project, chunking=PC)
    plain = search(client, auth, idx, "Saturn V rocket Apollo lunar orbit", top_k=3)
    child = plain["results"][0]
    assert child["parent_id"] and child["tokens"] <= 40 and "expanded" not in child
    big = search(client, auth, idx, "Saturn V rocket Apollo lunar orbit", top_k=3, expand="parent")
    top = big["results"][0]
    assert top["expanded"] == "parent" and top["tokens"] > child["tokens"] and "Saturn V" in top["text"]
    assert "Saturn V" in top["child_text"] and len(top["text"]) > len(top["child_text"])
    assert top["matched_children"] >= 1
    ids = [r["parent_id"] for r in big["results"]]
    assert len(ids) == len(set(ids))  # several children of one parent collapse into one result
    assert [r["rank"] for r in big["results"]] == list(range(1, len(big["results"]) + 1))


def test_parent_expansion_is_harmless_on_an_index_without_parents(client, auth, project, docs):
    idx = build(client, auth, project)
    r = search(client, auth, idx, "Saturn V rocket", top_k=3, expand="parent")
    assert r["results"] and all(x.get("expanded") == "none" for x in r["results"])


def test_neighbor_expansion_wraps_the_hit_in_adjacent_chunks(client, auth, project, docs):
    idx = build(client, auth, project)
    base = search(client, auth, idx, "Neil Armstrong and Buzz Aldrin walked on the Moon", top_k=1)["results"][0]
    wide = search(client, auth, idx, "Neil Armstrong and Buzz Aldrin walked on the Moon", top_k=1, expand="neighbors", neighbors=1)["results"][0]
    assert wide["expanded"] == "neighbors" and len(wide["neighbor_chunks"]) >= 2 and base["chunk_index"] in wide["neighbor_chunks"]
    assert wide["tokens"] > base["tokens"] and wide["child_text"] == base["text"] and base["text"] in wide["text"]


def test_expansion_flows_through_the_pipeline_and_answers(client, auth, project, docs, monkeypatch):
    from studio.config import settings
    from studio.generation import service as gen
    idx = build(client, auth, project, chunking=PC)
    r = client.post(f"/indexes/{idx['id']}/retrieve", headers=auth, json={"query": "Saturn V rocket Apollo", "expand": "parent", "top_k": 2}).json()
    assert r["results"][0]["expanded"] == "parent" and any(s["stage"] == "expand" for s in r["stages"])
    monkeypatch.setattr(settings, "azure_endpoint", "https://example.invalid")
    monkeypatch.setattr(settings, "azure_api_key", "not-a-real-key")
    monkeypatch.setattr(settings, "azure_chat_deployment", "fake-chat")
    seen = {}

    def fake(model, messages, temperature, max_output_tokens):
        seen["prompt"] = messages[1]["content"]
        return gen.ChatResult("The Saturn V [1].", 10, 5, 0, "stop", model)

    monkeypatch.setattr(gen, "chat_complete", fake)
    a = client.post(f"/indexes/{idx['id']}/answer", headers=auth, json={"question": "Which rocket?", "expand": "parent", "top_k": 2}).json()
    assert a["context_passages"][0]["text"] == r["results"][0]["text"] or len(seen["prompt"]) > 300  # the model saw the larger passage
    assert client.post(f"/indexes/{idx['id']}/search", headers=auth, json={"query": "x", "expand": "bogus"}).status_code == 422


def test_parent_child_validation_and_deletion(client, auth, project, docs):
    from sqlalchemy import select
    from studio import db as dbmod
    from studio.models import IndexParent
    for params, msg in [({"parent_size": 50, "child_size": 60}, "smaller than parent"), ({"child_size": 30, "child_overlap": 30}, "Child overlap")]:
        r = client.post(f"/projects/{project['id']}/indexes", headers=auth, json={"name": "x", "provider": "local_hash",
                        "chunking": {"method": "parent_child", "params": params}})
        assert r.status_code == 422 and msg in r.json()["detail"], r.text
    idx = build(client, auth, project, chunking=PC)
    with dbmod.SessionLocal() as db:
        assert db.scalars(select(IndexParent).where(IndexParent.index_id == idx["id"])).all()
    assert client.delete(f"/indexes/{idx['id']}", headers=auth).status_code == 204
    with dbmod.SessionLocal() as db:
        assert not db.scalars(select(IndexParent).where(IndexParent.index_id == idx["id"])).all()  # parents go with the index
