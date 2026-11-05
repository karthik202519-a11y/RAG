"""Vision descriptions of figure pages (fake vision model; tests never call Azure)."""
import pymupdf
import pytest

from conftest import make_pdf, register, upload, wait_for_job
from studio.config import settings
from studio.generation import service as gen
from test_indexes import build


def diagram_pdf() -> bytes:
    """Page 1: prose. Page 2: a drawn flowchart (many vector shapes) with a caption."""
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(72, 80, 540, 170), "Introduction to the requirements process described below.", fontsize=11)
    p = doc.new_page()
    for i in range(8):
        for j in range(8):
            p.draw_rect(pymupdf.Rect(60 + i * 55, 120 + j * 45, 100 + i * 55, 150 + j * 45), color=(0, 0, 0), width=0.8)
    p.insert_textbox(pymupdf.Rect(72, 60, 540, 100), "FIGURE 1 Requirements flow", fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def vision(monkeypatch):
    monkeypatch.setattr(settings, "azure_endpoint", "https://example.invalid")
    monkeypatch.setattr(settings, "azure_api_key", "not-a-real-key")
    monkeypatch.setattr(settings, "azure_vision_deployment", "fake-vision")
    calls = []

    def fake(model, prompt, image_png, max_output_tokens=700):
        calls.append({"model": model, "bytes": len(image_png), "starts_png": image_png[:4] == b"\x89PNG"})
        return gen.ChatResult("Flowchart: Stakeholder needs -> Requirements -> Design. Legend: dashed = feedback.",
                              900, 40, 0, "stop", model + "-served")

    monkeypatch.setattr(gen, "chat_vision", fake)
    return calls


@pytest.fixture()
def doc(client, auth, project):
    r = upload(client, auth, project["id"], "process.pdf", diagram_pdf()).json()["results"][0]
    assert wait_for_job(client, auth, r["job_id"])["status"] == "succeeded"
    return r["document_id"]


def figures(client, auth, doc):
    return client.get(f"/documents/{doc}/figures", headers=auth).json()


def test_drawn_page_is_flagged_and_listed(client, auth, doc):
    f = figures(client, auth, doc)
    assert [x["page"] for x in f["figures"]] == [2] and f["figures"][0]["has_image"] is True
    assert f["described"] == 0 and f["vision"]["configured"] is False


def test_describe_adds_searchable_text_and_checkpoints(client, auth, project, doc, vision):
    r = client.post(f"/documents/{doc}/describe-figures", headers=auth, json={})
    assert r.status_code == 202 and r.json()["pages"] == [2] and r.json()["model"] == "fake-vision"
    job = wait_for_job(client, auth, r.json()["job_id"])
    assert job["status"] == "succeeded"
    assert vision == [{"model": "fake-vision", "bytes": vision[0]["bytes"], "starts_png": True}]  # a real PNG was sent
    f = figures(client, auth, doc)
    assert f["described"] == 1 and f["figures"][0]["model"] == "fake-vision-served"
    assert f["figures"][0]["description"].startswith("[Figure description, page 2]")
    text = client.get(f"/documents/{doc}/content", headers=auth).json()["text"]
    assert "Stakeholder needs -> Requirements -> Design" in text
    v = client.get(f"/documents/{doc}", headers=auth).json()["latest_version"]
    assert v["token_count"] > 15  # counts were refreshed with the new text


def test_descriptions_are_indexed_and_found(client, auth, project, doc, vision):
    client.post(f"/documents/{doc}/describe-figures", headers=auth, json={})
    wait_for_job(client, auth, client.get(f"/projects/{project['id']}/ingestion-jobs", headers=auth).json()["items"][0]["id"])
    idx = build(client, auth, project, chunking={"method": "page_aware", "params": {"max_tokens": 0}})
    r = client.post(f"/indexes/{idx['id']}/search", headers=auth, json={"query": "stakeholder needs feedback legend dashed", "mode": "bm25"}).json()
    assert r["results"] and r["results"][0]["pages"] == [2] and "[Figure description" in r["results"][0]["text"]


def test_describe_is_idempotent_unless_forced(client, auth, project, doc, vision):
    def run(**body):
        j = client.post(f"/documents/{doc}/describe-figures", headers=auth, json=body).json()["job_id"]
        assert wait_for_job(client, auth, j)["status"] == "succeeded"
    run()
    run()  # nothing left to describe: no second paid call
    assert len(vision) == 1
    run(force=True)
    assert len(vision) == 2
    text = client.get(f"/documents/{doc}/content", headers=auth).json()["text"]
    assert text.count("[Figure description, page 2]") == 1  # replaced, not duplicated


def test_validation_and_errors(client, auth, project, doc):
    assert client.post(f"/documents/{doc}/describe-figures", headers=auth, json={}).status_code == 409  # not configured
    plain = upload(client, auth, project["id"], "plain.txt", b"just some text here").json()["results"][0]
    wait_for_job(client, auth, plain["job_id"])
    r = client.post(f"/documents/{plain['document_id']}/describe-figures", headers=auth, json={})
    assert r.status_code in (409, 422)


def test_validation_with_vision_configured(client, auth, project, doc, vision):
    plain = upload(client, auth, project["id"], "plain.txt", b"just some text here").json()["results"][0]
    wait_for_job(client, auth, plain["job_id"])
    r = client.post(f"/documents/{plain['document_id']}/describe-figures", headers=auth, json={})
    assert r.status_code == 422 and "No diagram" in r.json()["detail"]
    r = client.post(f"/documents/{doc}/describe-figures", headers=auth, json={"pages": [1]})
    assert r.status_code == 422 and "not flagged" in r.json()["detail"]


def test_vision_failure_retries_then_reports_without_losing_progress(client, auth, project, doc, vision, monkeypatch):
    def boom(*a, **k):
        raise gen.GenerationError("Vision request failed (APIConnectionError).")
    monkeypatch.setattr(gen, "chat_vision", boom)
    j = client.post(f"/documents/{doc}/describe-figures", headers=auth, json={}).json()["job_id"]
    job = wait_for_job(client, auth, j)
    assert job["status"] == "dead" and "Vision request failed" in job["error"] and job["attempts"] == 3
    assert figures(client, auth, doc)["described"] == 0 and "not-a-real-key" not in job["error"]


def test_figures_respect_access_control(client, auth, doc, vision):
    eve = register(client, "eve@example.com", "Eve")
    assert client.get(f"/documents/{doc}/figures", headers=eve).status_code == 404
    assert client.post(f"/documents/{doc}/describe-figures", headers=eve, json={}).status_code == 404
