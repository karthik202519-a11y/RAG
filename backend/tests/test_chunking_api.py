from conftest import upload, wait_for_job


def ready_doc(client, auth, project, sample_pdf):
    r = upload(client, auth, project["id"], "science.pdf", sample_pdf).json()["results"][0]
    assert wait_for_job(client, auth, r["job_id"])["status"] == "succeeded"
    return r["document_id"]


def test_catalog_lists_methods_with_honest_status(client, future_method):
    cat = client.get("/method-catalog", params={"category": "chunking"}).json()
    ids = {m["id"]: m for m in cat["items"]}
    assert ids["fixed_token"]["status"] == "implemented" and ids["fixed_token"]["params"]
    assert ids["future_method"]["status"] == "unavailable" and ids["future_method"]["unavailable_reason"]
    assert ids["topic_based"]["status"] == "implemented" and ids["adaptive"]["status"] == "implemented"
    assert ids["semantic"]["status"] == "implemented" and ids["semantic"]["params"]
    assert cat["counts"]["implemented"] + cat["counts"]["requires_credentials"] == 23 and cat["counts"]["unavailable"] == 1
    assert client.get("/method-catalog/recursive").json()["advantages"]
    assert client.get("/method-catalog/nope").status_code == 404
    only = client.get("/method-catalog", params={"category": "parser", "status": "implemented"}).json()["items"]
    assert {m["id"] for m in only} >= {"pymupdf", "plain_text", "markdown", "python_docx", "html_bs4", "csv", "json"}


def test_preview_compares_methods_on_the_same_text(client, auth, project, sample_pdf):
    doc = ready_doc(client, auth, project, sample_pdf)
    r = client.post("/chunking-preview", headers=auth, json={"document_id": doc, "configs": [
        {"method": "fixed_token", "params": {"size": 40, "overlap": 0}, "label": "A"},
        {"method": "fixed_token", "params": {"size": 40, "overlap": 20}, "label": "B"},
        {"method": "sentence", "params": {"size": 60, "overlap_sentences": 0}, "label": "C"}]}).json()
    a, b, c = r["results"]
    assert [x["label"] for x in r["results"]] == ["A", "B", "C"]
    assert b["stats"]["chunk_count"] > a["stats"]["chunk_count"]
    assert b["stats"]["duplication_ratio"] > a["stats"]["duplication_ratio"]
    assert c["stats"]["mid_sentence_pct"] <= a["stats"]["mid_sentence_pct"]
    for res in r["results"]:  # spans index into the single shared doc_text
        for ch in res["chunks"]:
            text = ch["text"] if ch["text"] is not None else r["doc_text"][ch["start"]:ch["end"]]
            assert text.strip() and ch["tokens"] > 0
    assert "Apollo" in r["doc_text"]


def test_preview_page_range(client, auth, project, sample_pdf):
    doc = ready_doc(client, auth, project, sample_pdf)
    r = client.post("/chunking-preview", headers=auth, json={
        "document_id": doc, "page_from": 2, "page_to": 2,
        "configs": [{"method": "page_aware", "params": {"max_tokens": 0}}]}).json()
    assert "Apollo" in r["doc_text"] and "Photosynthesis" not in r["doc_text"]
    assert all(c["pages"] == [2] for c in r["results"][0]["chunks"])
    bad = client.post("/chunking-preview", headers=auth, json={
        "document_id": doc, "page_from": 3, "page_to": 2, "configs": [{"method": "fixed_token"}]})
    assert bad.status_code == 422
    assert client.post("/chunking-preview", headers=auth, json={
        "document_id": doc, "page_from": 50, "configs": [{"method": "fixed_token"}]}).status_code == 422


def test_preview_validation_errors_are_422_with_a_reason(client, auth, project, sample_pdf, future_method):
    doc = ready_doc(client, auth, project, sample_pdf)

    def go(cfg):
        return client.post("/chunking-preview", headers=auth, json={"document_id": doc, "configs": [cfg]})

    r = go({"method": "fixed_token", "params": {"size": 50, "overlap": 50}})
    assert r.status_code == 422 and "Overlap" in r.json()["detail"]
    r = go({"method": "future_method"})
    assert r.status_code == 422 and "not available" in r.json()["detail"]
    assert go({"method": "nonsense"}).status_code == 422
    assert go({"method": "fixed_token", "params": {"size": "huge"}}).status_code == 422
    assert client.post("/chunking-preview", headers=auth, json={"document_id": doc, "configs": []}).status_code == 422


def test_preview_requires_ready_document_and_access(client, auth, project):
    from conftest import register
    r = upload(client, auth, project["id"], "x.txt", b"hello world " * 50).json()["results"][0]
    wait_for_job(client, auth, r["job_id"])
    body = {"document_id": r["document_id"], "configs": [{"method": "fixed_token"}]}
    assert client.post("/chunking-preview", headers=auth, json=body).status_code == 200
    assert client.post("/chunking-preview", json=body).status_code == 401
    assert client.post("/chunking-preview", headers=register(client, "eve@example.com", "Eve"), json=body).status_code == 404
