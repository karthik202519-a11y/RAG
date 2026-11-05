"""Figure crops, CLIP page search (real local model) and chart-to-table (fake vision model)."""
import pymupdf
import pytest

from conftest import upload, wait_for_job
from studio import multimodal
from studio.config import settings
from studio.generation import service as gen
from test_generation import configured  # noqa: F401


def figure_pdf() -> bytes:
    doc = pymupdf.open()
    for word in ("ROCKET", "FLOWER"):
        p = doc.new_page()
        p.insert_text((60, 300), word, fontsize=110)
    p = doc.new_page()
    p.insert_text((60, 60), "Figure 3 sales by quarter", fontsize=12)
    for i, h in enumerate((80, 140, 110, 180)):           # a bar chart made of vector rectangles
        p.draw_rect(pymupdf.Rect(80 + i * 60, 400 - h, 120 + i * 60, 400), color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
    p.draw_line((70, 400), (330, 400))
    png = pymupdf.open().new_page().get_pixmap(dpi=20).tobytes("png")  # a tiny blank picture: below the size threshold
    p.insert_image(pymupdf.Rect(400, 100, 410, 110), stream=png)
    p.insert_text((60, 700), "Body text under the chart.", fontsize=10)
    return doc.tobytes()


def test_find_figures_groups_the_bars_into_one_region_and_ignores_tiny_pictures():
    with pymupdf.open(stream=figure_pdf(), filetype="pdf") as doc:
        assert multimodal.find_figures(doc[0]) == []                       # a page of big text has no figure
        figs = multimodal.find_figures(doc[2])
    assert len(figs) == 1 and figs[0]["kind"] == "drawing"
    x0, y0, x1, y1 = figs[0]["bbox"]
    assert x0 <= 85 and x1 >= 310 and y0 <= 230 and y1 >= 395
    png = multimodal.render_crop(figure_pdf(), 3, 1)
    assert png[:4] == b"\x89PNG"


def test_crop_endpoints_and_validation(client, auth, project):
    res = upload(client, auth, project["id"], "fig.pdf", figure_pdf()).json()["results"][0]
    assert wait_for_job(client, auth, res["job_id"])["status"] == "succeeded"
    doc = res["document_id"]
    figs = client.get(f"/documents/{doc}/figure-crops", params={"page": 3}, headers=auth).json()["figures"]
    assert len(figs) == 1
    img = client.get(figs[0]["image_url"], headers=auth)
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert client.get(f"/documents/{doc}/figure-crops/3/9.png", headers=auth).status_code == 404
    assert client.get(f"/documents/{doc}/figure-crops", params={"page": 99}, headers=auth).status_code == 422
    assert client.get(f"/documents/{doc}/render/1.png", headers=auth).status_code == 200


def test_chart_to_table_uses_the_vision_model_and_warns(client, auth, project, configured, monkeypatch):  # noqa: F811
    monkeypatch.setattr(settings, "azure_vision_deployment", "fake-vision")
    seen = {}

    def fake_vision(model, prompt, png, max_tokens=700):
        seen.update(model=model, prompt=prompt, png=png[:4])
        return gen.ChatResult("Sales by quarter\n\n| Quarter | Sales |\n|---|---|\n| Q1 | 80 |", 500, 40, 0, "stop", model)
    monkeypatch.setattr(gen, "chat_vision", fake_vision)
    res = upload(client, auth, project["id"], "fig.pdf", figure_pdf()).json()["results"][0]
    wait_for_job(client, auth, res["job_id"])
    r = client.post(f"/documents/{res['document_id']}/chart-to-table", headers=auth, json={"page": 3, "figure": 1})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["is_chart"] and "| Q1 | 80 |" in j["markdown"] and "read from the image by a model" in j["warning"]
    assert seen["png"] == b"\x89PNG" and "Markdown table" in seen["prompt"]
    assert client.post(f"/documents/{res['document_id']}/chart-to-table", headers=auth, json={"page": 3, "figure": 7}).status_code == 404


def test_clip_finds_pages_by_what_they_show(client, auth, project):
    ok, _ = multimodal.clip_status()
    if not ok:
        pytest.skip("fastembed not installed")
    res = upload(client, auth, project["id"], "fig.pdf", figure_pdf()).json()["results"][0]
    wait_for_job(client, auth, res["job_id"])
    doc = res["document_id"]
    assert client.post(f"/documents/{doc}/page-images/search", headers=auth, json={"query": "ROCKET"}).status_code == 409
    job = client.post(f"/documents/{doc}/page-images", headers=auth).json()
    assert wait_for_job(client, auth, job["job_id"], timeout=900)["status"] == "succeeded"
    assert client.get(f"/documents/{doc}/page-images", headers=auth).json()["indexed_pages"] == 3
    top = lambda q: client.post(f"/documents/{doc}/page-images/search", headers=auth, json={"query": q, "top_k": 1}).json()["results"][0]["page"]  # noqa: E731
    assert top("the word ROCKET written in large letters") == 1
    assert top("the word FLOWER written in large letters") == 2
