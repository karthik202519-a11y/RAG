import pymupdf

from studio.parsing.compare import compare_pdf, similarity


def _pdf(text: str, pages: int = 2) -> bytes:
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 100), f"{text} page {i + 1}")
    return doc.tobytes()


def test_compare_runs_all_parsers_and_agrees_on_clean_text():
    out = compare_pdf(_pdf("Systems engineering handbook"), [1, 2], ["pymupdf", "pdfplumber", "pypdf"])
    assert [r["parser"] for r in out["results"]] == ["pymupdf", "pdfplumber", "pypdf"]
    assert all(r["error"] is None and len(r["pages"]) == 2 for r in out["results"])
    assert all("handbook" in r["pages"][0]["text"].lower() for r in out["results"])
    assert out["results"][2]["pages"][0]["tables"] is None  # pypdf cannot detect tables
    assert all(a["mean"] and a["mean"] > 0.9 for a in out["agreement"])


def test_similarity_edge_cases():
    assert similarity("", "") == 1.0
    assert similarity("alpha beta", "gamma delta") == 0.0


def test_endpoint(client, auth, project):
    from test_ingestion import ingest
    res = ingest(client, auth, project["id"], "lab.pdf", _pdf("Parser lab"))
    doc_id = res["document_id"]
    r = client.post(f"/documents/{doc_id}/parser-compare", headers=auth, json={"pages": [1]})
    assert r.status_code == 200, r.text
    assert len(r.json()["results"]) == 3
    assert client.post(f"/documents/{doc_id}/parser-compare", headers=auth, json={"pages": [99]}).status_code == 422
    assert client.post(f"/documents/{doc_id}/parser-compare", headers=auth, json={"pages": [1], "parsers": ["nope"]}).status_code == 422
