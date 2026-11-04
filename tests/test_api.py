"""End-to-end API tests: upload -> extract -> chunk -> embed -> search, plus error cases."""
from conftest import make_pdf


def upload(client, pdf: bytes, name="sample.pdf", **params):
    return client.post("/documents", files={"file": (name, pdf, "application/pdf")}, params=params)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["embedding_backend"] == "fake"


def test_upload_extracts_pages(client, sample_pdf):
    r = upload(client, sample_pdf)
    assert r.status_code == 201
    meta = r.json()
    assert meta["extracted"] is True and meta["pages"] == 3 and meta["created"] is True
    assert len(meta["doc_id"]) == 12


def test_upload_same_file_is_idempotent(client, sample_pdf):
    a = upload(client, sample_pdf).json()
    b = upload(client, sample_pdf, name="renamed.pdf").json()
    assert a["doc_id"] == b["doc_id"] and b["created"] is False
    assert len(client.get("/documents").json()) == 1


def test_rejects_non_pdf(client):
    r = client.post("/documents", files={"file": ("x.pdf", b"not a pdf at all", "application/pdf")})
    assert r.status_code == 415


def test_unknown_and_malformed_ids_are_404(client):
    assert client.get("/documents/aaaaaaaaaaaa").status_code == 404
    assert client.get("/documents/..%2F..%2Fetc").status_code == 404
    assert client.delete("/documents/not-an-id").status_code == 404


def test_pipeline_order_is_enforced(client, sample_pdf):
    doc = upload(client, sample_pdf, extract="false").json()
    d = doc["doc_id"]
    assert doc["extracted"] is False
    assert client.post(f"/documents/{d}/chunk", json={"size": 200}).status_code == 409
    assert client.get(f"/documents/{d}/pages/1").status_code == 409
    assert client.post(f"/documents/{d}/extract").status_code == 200
    assert client.post(f"/documents/{d}/embed", json={"size": 200}).status_code == 409  # not chunked yet
    assert client.post(f"/documents/{d}/chunk", json={"size": 200}).status_code == 200
    assert client.post(f"/documents/{d}/embed", json={"size": 200}).status_code == 200


def test_chunk_and_list(client, sample_pdf):
    d = upload(client, sample_pdf).json()["doc_id"]
    r = client.post(f"/documents/{d}/chunk", json={"size": 100}).json()
    assert r["chunks"] >= 1 and r["table"] == 0
    listed = client.get(f"/documents/{d}/chunks", params={"size": 100, "limit": 2}).json()
    assert listed["total"] == r["chunks"] and len(listed["chunks"]) <= 2
    c = listed["chunks"][0]
    assert c["doc_id"] == d and c["pages"] and "text" in c
    assert client.get(f"/documents/{d}/chunks", params={"size": 999}).status_code == 409


def test_chunk_size_validation(client, sample_pdf):
    d = upload(client, sample_pdf).json()["doc_id"]
    assert client.post(f"/documents/{d}/chunk", json={"size": 5}).status_code == 422
    assert client.post(f"/documents/{d}/chunk", json={"size": 99999}).status_code == 422


def test_search_finds_the_right_page_with_citation(client, sample_pdf):
    d = upload(client, sample_pdf).json()["doc_id"]
    assert client.post(f"/documents/{d}/process", json={"size": 60}).status_code == 200
    r = client.post("/search", json={"query": "Which rocket carried astronauts to the Moon?",
                                     "chunk_size": 60, "top_k": 3})
    assert r.status_code == 200
    top = r.json()["results"][0]
    # Chunks may span a page boundary (block-level overlap), so check membership.
    assert 2 in top["pages"] and top["rank"] == 1 and top["doc_id"] == d
    assert top["citation"].startswith("sample.pdf, PDF p")
    assert "Apollo" in top["text"]
    r = client.post("/search", json={"query": "How do SQL indexes work in a database?",
                                     "chunk_size": 60, "top_k": 1})
    assert 3 in r.json()["results"][0]["pages"]


def test_search_across_documents_and_filtering(client, sample_pdf):
    other = make_pdf([["Volcanoes erupt when magma rises through cracks in the crust."]])
    a = upload(client, sample_pdf, name="a.pdf").json()["doc_id"]
    b = upload(client, other, name="b.pdf").json()["doc_id"]
    for d in (a, b):
        client.post(f"/documents/{d}/process", json={"size": 60})
    r = client.post("/search", json={"query": "magma volcano eruption", "chunk_size": 60}).json()
    assert r["results"][0]["doc_id"] == b and set(r["searched"]) == {a, b}
    r = client.post("/search", json={"query": "magma volcano eruption", "chunk_size": 60, "doc_ids": [a]}).json()
    assert all(x["doc_id"] == a for x in r["results"])


def test_search_reports_documents_that_are_not_indexed(client, sample_pdf):
    d = upload(client, sample_pdf).json()["doc_id"]  # extracted but never chunked/embedded
    r = client.post("/search", json={"query": "anything", "chunk_size": 400}).json()
    assert r["results"] == [] and r["skipped"][0]["doc_id"] == d


def test_search_unknown_doc_id(client):
    r = client.post("/search", json={"query": "x", "doc_ids": ["aaaaaaaaaaaa"]})
    assert r.status_code == 404


def test_rechunk_invalidates_vectors(client, sample_pdf):
    d = upload(client, sample_pdf).json()["doc_id"]
    client.post(f"/documents/{d}/process", json={"size": 60})
    assert "60" in client.get(f"/documents/{d}").json()["embedded"]
    client.post(f"/documents/{d}/chunk", json={"size": 60})
    assert "60" not in client.get(f"/documents/{d}").json()["embedded"]


def test_delete_removes_everything(client, sample_pdf):
    d = upload(client, sample_pdf).json()["doc_id"]
    assert client.delete(f"/documents/{d}").status_code == 204
    assert client.get(f"/documents/{d}").status_code == 404
    assert client.get("/documents").json() == []


def test_azure_backend_without_config_returns_503(client, sample_pdf, monkeypatch):
    from app import config
    d = upload(client, sample_pdf).json()["doc_id"]
    client.post(f"/documents/{d}/chunk", json={"size": 60})
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "azure")
    monkeypatch.setattr(config, "AZURE_API_KEY", "")
    r = client.post(f"/documents/{d}/embed", json={"size": 60})
    assert r.status_code == 503 and "not configured" in r.json()["detail"]
