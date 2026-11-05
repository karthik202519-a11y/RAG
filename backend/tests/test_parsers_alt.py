"""Alternative parsers: pdfplumber tables, local OCR on scanned pages and images, Azure Document Intelligence with a recorded response."""
import json

import pymupdf
import pytest

from conftest import wait_for_job
from studio.config import settings
from studio.parsing import alt, parse_file, parser_options


def ruled_table_pdf(tmp_path) -> bytes:
    doc = pymupdf.open()
    pg = doc.new_page()
    pg.insert_text((72, 80), "Review criteria", fontsize=14)
    rows = [["Review", "Entrance", "Success"], ["PDR", "Baseline ready", "Design meets requirements"], ["CDR", "Design complete", "Ready to build"]]
    x0, y0, cw, rh = 72, 110, 150, 24
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            pg.draw_rect(rect, color=(0, 0, 0), width=0.8)
            pg.insert_text((rect.x0 + 4, rect.y0 + 16), cell, fontsize=9)
    return doc.tobytes()


def scanned_pdf_and_png() -> tuple[bytes, bytes]:
    src = pymupdf.open()
    p = src.new_page()
    p.insert_text((72, 100), "Scanned Handbook Page", fontsize=22)
    p.insert_text((72, 150), "The Preliminary Design Review demonstrates the design meets requirements.", fontsize=11)
    pix = p.get_pixmap(dpi=150)
    png = pix.tobytes("png")
    scanned = pymupdf.open()
    sp = scanned.new_page(width=p.rect.width, height=p.rect.height)
    sp.insert_image(sp.rect, stream=png)  # a page that is only a picture: no text layer
    return scanned.tobytes(), png


def test_pdfplumber_extracts_the_ruled_table_as_markdown(tmp_path):
    f = tmp_path / "t.pdf"
    f.write_bytes(ruled_table_pdf(tmp_path))
    pid, pages = parse_file(f, "pdf", "t.pdf", parser="pdfplumber")
    assert pid == "pdfplumber"
    tables = [i for p in pages for i in p["items"] if i["type"] == "table"]
    assert len(tables) == 1 and "| PDR | Baseline ready | Design meets requirements |" in tables[0]["markdown"]
    text = " ".join(i["text"] for p in pages for i in p["items"] if i["type"] == "text")
    assert "Review criteria" in text and "Baseline ready" not in text  # table cells are not repeated as prose


def test_ocr_reads_a_scanned_pdf_that_has_no_text_layer(tmp_path):
    pdf, png = scanned_pdf_and_png()
    f = tmp_path / "scan.pdf"
    f.write_bytes(pdf)
    default_id, default_pages = parse_file(f, "pdf", "scan.pdf")
    assert default_id == "pymupdf" and not any(i["type"] == "text" and i["text"].strip() for p in default_pages for i in p["items"])
    pid, pages = parse_file(f, "pdf", "scan.pdf", parser="rapidocr")
    text = " ".join(i["text"] for p in pages for i in p["items"]).lower()
    assert pid == "rapidocr" and "scanned handbook page" in text and "preliminary design review" in text
    img = tmp_path / "scan.png"
    img.write_bytes(png)
    pid, pages = parse_file(img, "png", "scan.png")
    assert pid == "rapidocr" and "design meets requirements" in " ".join(i["text"] for i in pages[0]["items"]).lower()


def test_parser_choice_is_validated():
    assert parser_options("pdf")[:2] == ["pymupdf", "pdfplumber"] and parser_options("txt") == []
    with pytest.raises(ValueError, match="cannot read"):
        parse_file(__import__("pathlib").Path("x.txt"), "txt", "x", parser="pdfplumber")


def test_upload_with_a_parser_choice_and_image_upload(client, auth, project, tmp_path):
    from conftest import upload
    pdf, png = scanned_pdf_and_png()
    r = upload(client, auth, project["id"], "scan.pdf", pdf, parser="rapidocr")
    assert r.status_code == 201, r.text
    res = r.json()["results"][0]
    job = wait_for_job(client, auth, res["job_id"], timeout=120)
    assert job["status"] == "succeeded", job
    content = client.get(f"/documents/{res['document_id']}/content", headers=auth).json()
    assert "preliminary design review" in json.dumps(content).lower()
    assert client.get(f"/documents/{res['document_id']}", headers=auth).json()["latest_version"]["parser"] == "rapidocr"
    bad = upload(client, auth, project["id"], "x.txt", b"hello", parser="pdfplumber").json()["results"][0]
    assert bad["status"] == "rejected" and "cannot read" in bad["error"]
    img = upload(client, auth, project["id"], "scan.png", png).json()["results"][0]
    assert img["status"] == "queued"
    assert wait_for_job(client, auth, img["job_id"], timeout=120)["status"] == "succeeded"
    fake = upload(client, auth, project["id"], "fake.png", b"not an image at all").json()["results"][0]
    assert fake["status"] == "rejected"


class FakeHttp:
    """Records the calls and returns what Azure's prebuilt-layout API returns, in the shape documented for api-version 2024-11-30."""
    def __init__(self):
        self.calls = []

    def post(self, url, headers=None, content=None, timeout=None):
        self.calls.append(("POST", url))

        class R:
            status_code = 202
            headers = {"Operation-Location": "https://example.invalid/operations/1"}
        return R()

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url))
        poly = lambda y: [10, y, 300, y, 300, y + 12, 10, y + 12]  # noqa: E731
        body = {"status": "succeeded", "analyzeResult": {
            "pages": [{"pageNumber": 1, "lines": [{"content": "Review criteria", "polygon": poly(10)}, {"content": "Applies to all projects.", "polygon": poly(60)}]}],
            "tables": [{"rowCount": 2, "columnCount": 2, "boundingRegions": [{"pageNumber": 1}],
                        "cells": [{"rowIndex": 0, "columnIndex": 0, "content": "Review"}, {"rowIndex": 0, "columnIndex": 1, "content": "Success"},
                                  {"rowIndex": 1, "columnIndex": 0, "content": "PDR"}, {"rowIndex": 1, "columnIndex": 1, "content": "Meets requirements"}]}]}}

        class R:
            def json(self_inner):
                return body
        return R()


def test_azure_document_intelligence_maps_lines_and_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "azure_di_endpoint", "https://example.invalid")
    monkeypatch.setattr(settings, "azure_di_key", "k")
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    http = FakeHttp()
    pages = alt.extract_azure_di(f, "a.pdf", http=http)
    assert [c[0] for c in http.calls] == ["POST", "GET"] and "prebuilt-layout:analyze" in http.calls[0][1]
    items = pages[0]["items"]
    assert [i["text"] for i in items if i["type"] == "text"] == ["Review criteria", "Applies to all projects."]
    assert "| PDR | Meets requirements |" in next(i for i in items if i["type"] == "table")["markdown"]
    monkeypatch.setattr(settings, "azure_di_key", "")
    with pytest.raises(ValueError, match="AZURE_DOCUMENT_INTELLIGENCE"):
        alt.extract_azure_di(f, "a.pdf", http=http)
