"""Step 9: build a local, self-contained viewer of the whole pipeline.

Usage:  python scripts/09_build_viewer.py
Output: D:\\RAG-I\\viewer\\data.js  + viewer\\pages\\<doc>_<page>.jpg  (open viewer\\index.html)

For every page we keep: the page image, the extracted items, and the chunks that
cover it. For every chunk we keep text, metadata and its vector (int8, base64) so
the browser can show a vector fingerprint and nearest neighbours.
"""
import base64
import json
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "viewer"
DOCS = {
    "nasa": {
        "label": "NASA Systems Engineering Handbook",
        "pdf": "documents/nasa_systems_engineering_handbook_0.pdf",
        "extracted": "output/extracted/nasa_systems_engineering_handbook_0.json",
        "chunks": "output/chunks/chunks_400.json",
        "vectors": "output/embeddings/chunks_400.npy",
    },
    "ipcc": {
        "label": "IPCC AR6 Synthesis Report",
        "pdf": "documents/IPCC_AR6_SYR_FullVolume.pdf",
        "extracted": "output/extracted_ipcc/IPCC_AR6_SYR_FullVolume.json",
        "chunks": "output/chunks_ipcc/chunks_400.json",
        "vectors": "output/embeddings/chunks_ipcc_chunks_400.npy",
    },
}
BINS = 96  # the "fingerprint" strip: 1536 dims averaged into 96 cells

SHOWCASE = [
    {"doc": "nasa", "page": 150, "title": "Prose page",
     "note": "Paragraphs become text blocks, then blocks are grouped up to ~400 tokens. Known flaw: a boxed heading ran into its first sentence ('Key Concepts in Risk Management Risk: ...')."},
    {"doc": "nasa", "page": 17, "title": "Simple table",
     "note": "A clean two-column table is detected and kept whole, as Markdown, in its own chunk with its section title attached."},
    {"doc": "nasa", "page": 29, "title": "Large table (22 x 10 grid)",
     "note": "The whole grid is ONE chunk (~930 tokens) and therefore ONE vector. Row/column structure survives as Markdown, but a question about a single cell has to match a vector averaged over the entire grid."},
    {"doc": "nasa", "page": 125, "title": "Flowchart",
     "note": "Only the text inside the boxes is extracted. The arrows, order and the 'from / to' columns are drawn shapes, so they are not in any chunk. The page is flagged needs_vision."},
    {"doc": "nasa", "page": 68, "title": "Requirements flowdown diagram",
     "note": "Box labels survive as loose text lines; which box feeds which is carried by arrows and is lost. A vision model would have to describe it."},
    {"doc": "nasa", "page": 28, "title": "Figure misread as a table",
     "note": "A life-cycle FIGURE was detected as a table and came out with words cut apart ('Approv / val', 'Peer Revie / ews'). Red-triangle review markers are lost. A wrong chunk like this is worse than no chunk."},
    {"doc": "ipcc", "page": 65, "title": "Scatter plot + dot-matrix chart",
     "note": "The chart title, axis names and the annotation text are extracted. The dots, their positions, and the dots/dashes/slashes in the matrix (which carry the actual data) are not."},
    {"doc": "ipcc", "page": 30, "title": "Colour-scaled world maps",
     "note": "Only the captions, panel titles and colour-scale numbers are text. The patterns on the maps are pixels, so no chunk or vector contains them."},
]


def main() -> None:
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    all_chunks, vec_rows, docs_out = [], [], {}

    for key, cfg in DOCS.items():
        pages = json.loads((ROOT / cfg["extracted"]).read_text(encoding="utf-8"))
        chunks = json.loads((ROOT / cfg["chunks"]).read_text(encoding="utf-8"))
        mat = np.load(ROOT / cfg["vectors"]).astype(np.float32)
        assert len(mat) == len(chunks), (key, len(mat), len(chunks))
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)

        base = len(all_chunks)
        by_page: dict[int, list[int]] = {}
        for i, c in enumerate(chunks):
            gid = base + i
            for p in c["pages"]:
                by_page.setdefault(p, []).append(gid)
            all_chunks.append({
                "g": gid, "doc": key, "cid": c["chunk_id"], "type": c["type"], "tok": c["tokens"],
                "pages": c["pages"], "printed": c["printed_pages"], "sec": c["section"],
                "text": c["text"], "head": [round(float(x), 4) for x in mat[i][:8]],
            })
        vec_rows.append(mat)

        # render page images once (small JPEGs; originals stay in documents/)
        with pymupdf.open(ROOT / cfg["pdf"]) as pdf:
            for page in pdf:
                f = OUT / "pages" / f"{key}_{page.number + 1}.jpg"
                if not f.exists():
                    f.write_bytes(page.get_pixmap(matrix=pymupdf.Matrix(0.9, 0.9)).tobytes("jpeg", jpg_quality=70))

        docs_out[key] = {"label": cfg["label"], "pages": [{
            "n": p["pdf_page"], "printed": p["printed_page"], "sec": p["section"],
            "vision": p["needs_vision"], "outside": p["drawings_outside_tables"],
            "items": [({"t": "text", "s": i["text"][:1200]} if i["type"] == "text"
                       else {"t": "table", "s": i["markdown"][:4500]}) for i in p["items"]],
            "chunks": by_page.get(p["pdf_page"], []),
        } for p in pages]}
        print(f"{key}: {len(pages)} pages, {len(chunks)} chunks")

    V = np.vstack(vec_rows)
    bins = V.reshape(len(V), BINS, -1).mean(axis=2)
    scale = 127 / np.percentile(np.abs(bins), 99)
    strips = np.clip(np.round(bins * scale), -127, 127).astype(int)
    for c, s in zip(all_chunks, strips):
        c["strip"] = s.tolist()

    q = np.zeros(V.shape, dtype=np.int8)
    for i, v in enumerate(V):
        q[i] = np.round(v / np.abs(v).max() * 127).astype(np.int8)  # cosine is scale-free

    data = {"dim": int(V.shape[1]), "docs": docs_out, "chunks": all_chunks,
            "showcase": SHOWCASE, "vecs": base64.b64encode(q.tobytes()).decode("ascii")}
    (OUT / "data.js").write_text("window.DATA = " + json.dumps(data, ensure_ascii=False) + ";", encoding="utf-8")
    print(f"chunks total {len(all_chunks)} | data.js {(OUT / 'data.js').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
