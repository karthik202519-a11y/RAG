"""Run several PDF parsers on the same pages so their output can be compared side by side.

Only the comparison lab uses pdfplumber and pypdf; ingestion stays on PyMuPDF. Everything here is measured on the
real file: text, table counts and timings are what each library returned, nothing is estimated.
"""
import io
import re
import time
from difflib import SequenceMatcher

import pymupdf

COMPARABLE = ("pymupdf", "pdfplumber", "pypdf")
MAX_PAGES = 12
_WORD = re.compile(r"\w+", re.UNICODE)


def _pymupdf(data: bytes, pages: list[int]) -> list[dict]:
    out = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for n in pages:
            t0 = time.perf_counter()
            page = doc[n - 1]
            text = page.get_text("text")
            tables = page.find_tables().tables
            out.append({"page": n, "text": text, "tables": len(tables), "seconds": time.perf_counter() - t0})
    return out


def _pdfplumber(data: bytes, pages: list[int]) -> list[dict]:
    import pdfplumber
    out = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for n in pages:
            t0 = time.perf_counter()
            page = pdf.pages[n - 1]
            text = page.extract_text() or ""
            tables = page.find_tables()
            out.append({"page": n, "text": text, "tables": len(tables), "seconds": time.perf_counter() - t0})
    return out


def _pypdf(data: bytes, pages: list[int]) -> list[dict]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    out = []
    for n in pages:
        t0 = time.perf_counter()
        text = reader.pages[n - 1].extract_text() or ""
        out.append({"page": n, "text": text, "tables": None, "seconds": time.perf_counter() - t0})  # pypdf has no table detector
    return out


RUNNERS = {"pymupdf": _pymupdf, "pdfplumber": _pdfplumber, "pypdf": _pypdf}


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def similarity(a: str, b: str) -> float:
    """Share of words two extractions agree on, in order (difflib ratio over word lists)."""
    wa, wb = _words(a), _words(b)
    if not wa and not wb:
        return 1.0
    return round(SequenceMatcher(None, wa, wb, autojunk=False).ratio(), 4)


def compare_pdf(data: bytes, pages: list[int], parsers: list[str]) -> dict:
    results: dict[str, dict] = {}
    for pid in parsers:
        t0 = time.perf_counter()
        try:
            per_page = RUNNERS[pid](data, pages)
            error = None
        except Exception as e:  # a parser failing on a page is a finding, not a crash
            per_page, error = [], f"{type(e).__name__}: {str(e)[:200]}"
        results[pid] = {"parser": pid, "error": error, "seconds": round(time.perf_counter() - t0, 4), "pages": per_page}
        for p in per_page:
            p["seconds"] = round(p["seconds"], 4)
            p["chars"], p["words"] = len(p["text"]), len(_words(p["text"]))
            p["empty"] = not p["text"].strip()
    agreement = []
    ids = [p for p in parsers if not results[p]["error"]]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            per = [similarity(pa["text"], pb["text"]) for pa, pb in zip(results[a]["pages"], results[b]["pages"])]
            agreement.append({"a": a, "b": b, "mean": round(sum(per) / len(per), 4) if per else None, "per_page": per})
    return {"pages": pages, "results": list(results.values()), "agreement": agreement}
