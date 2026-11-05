"""Alternative parsers an upload can choose instead of the default.

pdfplumber                   PDF text and ruled tables (found tables PyMuPDF missed on the NASA handbook)
rapidocr                     local OCR (ONNX) for scanned PDFs and images; no key, no server
azure_document_intelligence  Azure AI Document Intelligence 'prebuilt-layout' (REST); needs AZURE_DOCUMENT_INTELLIGENCE_* settings

Output has the same shape as the PyMuPDF parser, so chunking and indexing do not care which parser ran.
"""
import io
import re
import statistics
import time
from pathlib import Path

import numpy as np

from ..config import settings
from .pdf import clean
from .textlike import _md_table, _page

OCR_DPI = 200
_ocr = None


def _table_md(rows: list[list]) -> str:
    rows = [[("" if c is None else str(c)).replace("\n", " ").strip() for c in r] for r in rows if r and any(c for c in r)]
    return _md_table(rows) if rows else ""


# ----------------------------------------------------------------------------- pdfplumber
def extract_pdfplumber(path: Path, doc_name: str, progress=None, cancelled=None) -> list[dict]:
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for n, pg in enumerate(pdf.pages, 1):
            if cancelled and cancelled():
                raise InterruptedError("cancelled")
            tables = pg.find_tables()
            boxes = [t.bbox for t in tables]

            def outside(o, boxes=boxes):
                x0, top, x1, bottom = o.get("x0", 0), o.get("top", 0), o.get("x1", 0), o.get("bottom", 0)
                return not any(x0 >= b[0] - 1 and x1 <= b[2] + 1 and top >= b[1] - 1 and bottom <= b[3] + 1 for b in boxes)

            body = pg.filter(outside).extract_text() if boxes else pg.extract_text()
            items: list[dict] = []
            for block in re.split(r"\n\s*\n", body or ""):
                t = clean(block.replace("\n", " "))
                if t:
                    items.append({"type": "text", "text": t})
            if not items and body and body.strip():
                items.append({"type": "text", "text": clean(body.replace("\n", " "))})
            for t in tables:
                md = _table_md(t.extract())
                if md:
                    items.append({"type": "table", "bbox": [round(v) for v in t.bbox], "markdown": md})
            pages.append(_page(doc_name, n, items))
            if progress:
                progress(n, len(pdf.pages))
    return pages


# ----------------------------------------------------------------------------- OCR
def _engine():
    global _ocr
    if _ocr is None:
        import logging
        from rapidocr import RapidOCR
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        _ocr = RapidOCR()
    return _ocr


def ocr_image(img: np.ndarray) -> list[dict]:
    """Recognised text lines with their box, top to bottom: [{text, x0, y0, x1, y1, score}]."""
    r = _engine()(img)
    if r is None or r.txts is None:
        return []
    out = []
    for t, b, s in zip(r.txts, r.boxes, r.scores):
        xs, ys = [p[0] for p in b], [p[1] for p in b]
        out.append({"text": str(t), "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys), "score": float(s)})
    return sorted(out, key=lambda x: (x["y0"], x["x0"]))


def lines_to_paragraphs(lines: list[dict]) -> list[str]:
    """Join OCR lines into paragraphs: a vertical gap bigger than about one and a half line heights starts a new paragraph."""
    if not lines:
        return []
    h = statistics.median([ln["y1"] - ln["y0"] for ln in lines]) or 10
    paras, cur, prev = [], [], None
    for ln in lines:
        if prev is not None and ln["y0"] - prev["y1"] > 0.75 * h:
            paras.append(" ".join(cur))
            cur = []
        cur.append(ln["text"])
        prev = ln
    if cur:
        paras.append(" ".join(cur))
    return [clean(p) for p in paras if p.strip()]


def extract_ocr_pdf(path: Path, doc_name: str, progress=None, cancelled=None) -> list[dict]:
    import pymupdf
    pages = []
    with pymupdf.open(path) as doc:
        for n, pg in enumerate(doc, 1):
            if cancelled and cancelled():
                raise InterruptedError("cancelled")
            pix = pg.get_pixmap(dpi=OCR_DPI, colorspace=pymupdf.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            pages.append(_page(doc_name, n, [{"type": "text", "text": p} for p in lines_to_paragraphs(ocr_image(img))]))
            if progress:
                progress(n, len(doc))
    return pages


def extract_ocr_image(path: Path, doc_name: str) -> list[dict]:
    import pymupdf
    pix = pymupdf.Pixmap(str(path))
    if pix.alpha or pix.colorspace.n != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return [_page(doc_name, 1, [{"type": "text", "text": p} for p in lines_to_paragraphs(ocr_image(img))])]


# ----------------------------------------------------------------------------- Azure Document Intelligence
def azure_di_status() -> tuple[bool, str]:
    if settings.azure_di_endpoint and settings.azure_di_key:
        return True, "configured"
    return False, "Set in .env: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY"


def extract_azure_di(path: Path, doc_name: str, progress=None, cancelled=None, http=None) -> list[dict]:
    """POST the file to the prebuilt-layout model, poll the operation, and map pages (lines) and tables to our page shape."""
    import httpx
    ok, why = azure_di_status()
    if not ok:
        raise ValueError(why)
    http = http or httpx
    base = settings.azure_di_endpoint.rstrip("/")
    hdr = {"Ocp-Apim-Subscription-Key": settings.azure_di_key, "Content-Type": "application/octet-stream"}
    r = http.post(f"{base}/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30", headers=hdr,
                  content=path.read_bytes(), timeout=120)
    if r.status_code != 202:
        raise ValueError(f"Document Intelligence rejected the file (HTTP {r.status_code})")
    op = r.headers["Operation-Location"]
    for _ in range(120):
        if cancelled and cancelled():
            raise InterruptedError("cancelled")
        j = http.get(op, headers={"Ocp-Apim-Subscription-Key": settings.azure_di_key}, timeout=60).json()
        if j.get("status") == "succeeded":
            break
        if j.get("status") == "failed":
            raise ValueError("Document Intelligence analysis failed")
        time.sleep(1.5)
    else:
        raise ValueError("Document Intelligence timed out")
    res = j["analyzeResult"]
    by_page: dict[int, dict] = {p["pageNumber"]: {"lines": p.get("lines", []), "tables": []} for p in res.get("pages", [])}
    for t in res.get("tables", []):
        pn = (t.get("boundingRegions") or [{"pageNumber": 1}])[0]["pageNumber"]
        grid: dict[tuple[int, int], str] = {(c["rowIndex"], c["columnIndex"]): c.get("content", "") for c in t.get("cells", [])}
        rows = [[grid.get((r, c), "") for c in range(t["columnCount"])] for r in range(t["rowCount"])]
        by_page.setdefault(pn, {"lines": [], "tables": []})["tables"].append(_table_md(rows))
    pages = []
    for pn in sorted(by_page):
        paras = lines_to_paragraphs([{"text": ln["content"], "y0": ln["polygon"][1], "y1": ln["polygon"][5], "x0": ln["polygon"][0], "x1": ln["polygon"][2]}
                                     for ln in by_page[pn]["lines"] if len(ln.get("polygon", [])) >= 6])
        items = [{"type": "text", "text": p} for p in paras] + [{"type": "table", "bbox": None, "markdown": m} for m in by_page[pn]["tables"] if m]
        pages.append(_page(doc_name, pn, items))
    return pages
