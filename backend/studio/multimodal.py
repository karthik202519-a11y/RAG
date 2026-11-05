"""Visual content: figure crops, CLIP page-image search and chart-to-table.

Figure crops   embedded pictures and clusters of vector drawings found on a PDF page, cut out at 150 dpi
CLIP search    every page rendered and embedded with a CLIP image model; a text query is embedded with the paired CLIP text model, so
               "a flowchart of the review process" finds pages by what they look like. Local (ONNX), no key.
Chart to table a vision model reads a chart image and writes the values as a Markdown table, with a warning that values are read by a model
"""
import threading

import numpy as np
import pymupdf

from .config import settings
from .vectorstores import faiss_store

CROP_DPI = 150
MIN_SIDE_PT = 60          # ignore icons and rules smaller than this on either side
MIN_AREA_SHARE = 0.02     # a figure should cover at least this share of the page
_lock = threading.Lock()
_models: dict = {}


# ------------------------------------------------------------------------------------ crops
def _rect_union_merge(rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    rects = list(rects)
    changed = True
    while changed:
        changed = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                grown = pymupdf.Rect(a.x0 - 6, a.y0 - 6, a.x1 + 6, a.y1 + 6)
                if grown.intersects(b):
                    rects[i] = a | b
                    del rects[j]
                    changed = True
                    break
            if changed:
                break
    return rects


def find_figures(page: pymupdf.Page) -> list[dict]:
    """Candidate figure regions on one page: raster pictures and clusters of vector drawings, merged where they touch."""
    found: list[tuple[str, pymupdf.Rect]] = []
    for info in page.get_image_info():
        found.append(("image", pymupdf.Rect(info["bbox"])))
    try:
        found += [("drawing", pymupdf.Rect(r)) for r in page.cluster_drawings()]
    except Exception:
        pass
    page_area = page.rect.get_area()
    keep = []
    for kind, r in found:
        r = r & page.rect
        if r.is_empty or r.width < MIN_SIDE_PT or r.height < MIN_SIDE_PT or r.get_area() < MIN_AREA_SHARE * page_area:
            continue
        keep.append((kind, r))
    merged = _rect_union_merge([r for _, r in keep])
    kinds = {tuple(round(v) for v in r): k for k, r in keep}
    out = []
    for n, r in enumerate(sorted(merged, key=lambda r: (r.y0, r.x0)), 1):
        kind = next((k for k, o in keep if o.intersects(r)), "drawing")
        out.append({"n": n, "kind": kind, "bbox": [round(r.x0), round(r.y0), round(r.x1), round(r.y1)],
                    "width_pt": round(r.width), "height_pt": round(r.height), "share_of_page": round(r.get_area() / page_area, 3)})
    del kinds
    return out


def render_crop(pdf_bytes: bytes, page_no: int, n: int) -> bytes:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        figs = find_figures(doc[page_no - 1])
        fig = next((f for f in figs if f["n"] == n), None)
        if fig is None:
            raise KeyError(f"Page {page_no} has no figure {n}")
        return doc[page_no - 1].get_pixmap(dpi=CROP_DPI, clip=pymupdf.Rect(*fig["bbox"])).tobytes("png")


def render_page(pdf_bytes: bytes, page_no: int, dpi: int = 100) -> bytes:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")


# ------------------------------------------------------------------------------------ CLIP
def clip_status() -> tuple[bool, str]:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False, "pip install fastembed"
    return True, f"local models {settings.fastembed_clip_image} and {settings.fastembed_clip_text} (large download on first use)"


def _clip():
    with _lock:
        if "img" not in _models:
            from fastembed import ImageEmbedding, TextEmbedding
            cache = str(settings.data_dir / "models")
            _models["img"] = ImageEmbedding(settings.fastembed_clip_image, cache_dir=cache)
            _models["txt"] = TextEmbedding(settings.fastembed_clip_text, cache_dir=cache)
        return _models["img"], _models["txt"]


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def embed_images(pngs: list[bytes]) -> np.ndarray:
    import io

    from PIL import Image
    img, _ = _clip()
    ims = [Image.open(io.BytesIO(b)).convert("RGB") for b in pngs]
    return _unit(np.asarray(list(img.embed(ims)), dtype=np.float32))


def embed_text(q: str) -> np.ndarray:
    _, txt = _clip()
    return _unit(np.asarray(list(txt.embed([q])), dtype=np.float32))[0]


def collection(version_id: str) -> str:
    return f"pages_{version_id}"


def build_page_index(version_id: str, pdf_bytes: bytes, progress=None, cancelled=None, batch: int = 8) -> int:
    """Render and embed every page; returns the number of pages stored."""
    name = collection(version_id)
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = len(doc)
        faiss_store.create_collection(name, 512, "cosine")
        done = 0
        for start in range(0, total, batch):
            if cancelled and cancelled():
                raise InterruptedError("cancelled")
            idx = list(range(start, min(start + batch, total)))
            pngs = [doc[i].get_pixmap(dpi=72).tobytes("png") for i in idx]
            vecs = embed_images(pngs)
            faiss_store.upsert(name, [f"{i + 1:032x}" for i in idx], vecs, [{"document_id": version_id, "ctype": "page"} for _ in idx])
            done += len(idx)
            if progress:
                progress(done / total, f"page {done}/{total}")
    return done


def search_pages(version_id: str, query: str, top_k: int = 5) -> list[dict]:
    hits = faiss_store.search(collection(version_id), embed_text(query), top_k)
    return [{"page": int(h.id, 16), "score": round(h.score, 4)} for h in hits]


# ------------------------------------------------------------------------------------ chart to table
CHART_PROMPT = (
    "This image may be a chart, graph or table. If it shows data, write the data as ONE Markdown table: the first column is the category or "
    "x value, then one column per series, with units in the headers. Give a one-line title above the table. Only include values you can read "
    "with reasonable confidence; write '?' for a value you cannot read. If the image is not a chart or table, reply exactly: NOT A CHART."
)
