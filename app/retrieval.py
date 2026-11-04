"""Dense retrieval: cosine similarity between the query vector and chunk vectors."""
import numpy as np

from . import embeddings
from .store import NotFound, Store


def citation(filename: str, chunk: dict) -> str:
    pages = chunk["pages"]
    printed = [p for p in chunk.get("printed_pages", []) if p]
    span = f"p.{pages[0]}" if len(pages) == 1 else f"pp.{pages[0]}-{pages[-1]}"
    s = f"{filename}, PDF {span}"
    if printed:
        s += f" (printed {printed[0]}" + (f"-{printed[-1]}" if len(printed) > 1 else "") + ")"
    return s


def search(store: Store, query: str, top_k: int, size: int,
           doc_ids: list[str] | None = None, types: list[str] | None = None) -> dict:
    """Search the chunks of the given documents (default: all) at one chunk size.

    Documents that have no vectors at `size` are reported in `skipped`, not hidden.
    """
    docs = [store.get(d) for d in doc_ids] if doc_ids else store.list()
    searched, skipped, chunks, mats = [], [], [], []
    for m in docs:
        try:
            vec = store.load_vectors(m["doc_id"], size)
            ch = store.load_chunks(m["doc_id"], size)
        except (FileNotFoundError, NotFound):
            skipped.append({"doc_id": m["doc_id"], "filename": m["filename"],
                            "reason": f"not chunked and embedded at size {size}"})
            continue
        if len(vec) != len(ch):
            skipped.append({"doc_id": m["doc_id"], "filename": m["filename"],
                            "reason": "vectors out of date; re-run embed"})
            continue
        for c in ch:
            c["_doc"] = m
        chunks += ch
        mats.append(vec)
        searched.append(m["doc_id"])

    if not chunks:
        return {"query": query, "chunk_size": size, "results": [], "searched": searched, "skipped": skipped}

    qvec, _ = embeddings.embed_texts([query])
    mat = np.vstack(mats)
    if mat.shape[1] != qvec.shape[1]:
        raise ValueError(f"Query vector has {qvec.shape[1]} dims but stored vectors have {mat.shape[1]}; "
                         "documents were embedded with a different backend. Re-run embed.")
    sims = mat @ qvec[0]
    if types:
        sims = np.where([c["type"] in types for c in chunks], sims, -np.inf)
    order = [i for i in np.argsort(-sims)[:top_k] if np.isfinite(sims[i])]
    results = []
    for rank, i in enumerate(order, start=1):
        c, m = chunks[i], chunks[i]["_doc"]
        results.append({
            "rank": rank,
            "score": round(float(sims[i]), 4),
            "doc_id": m["doc_id"],
            "filename": m["filename"],
            "chunk_id": c["chunk_id"],
            "type": c["type"],
            "section": c.get("section"),
            "pages": c["pages"],
            "printed_pages": c.get("printed_pages", []),
            "tokens": c["tokens"],
            "citation": citation(m["filename"], c),
            "text": c["text"],
        })
    return {"query": query, "chunk_size": size, "results": results, "searched": searched, "skipped": skipped}
