"""Retrieval over a built index: dense (vector), BM25 (keyword) and hybrid (reciprocal rank fusion).

Every result reports what each retriever contributed (rank and score), so ranking is explainable from
observable numbers rather than a generated rationale.
"""
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..embeddings import embed_texts
from ..embeddings.providers import ProviderError
from ..models import Index, IndexChunk
from ..vectorstores import qdrant_store

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
MODES = ("dense", "bm25", "hybrid")


class DimensionMismatch(ProviderError):
    """Query vector and index vectors have different sizes; searching would be meaningless."""


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class _Bm25:
    bm25: BM25Okapi
    ids: list[str]
    doc_ids: list[str]
    types: list[str]
    tokens: list[set[str]] = field(default_factory=list)


_cache: dict[str, _Bm25] = {}
_lock = threading.Lock()


def drop_bm25(index_id: str) -> None:
    with _lock:
        _cache.pop(index_id, None)


def _bm25_for(db: Session, index_id: str) -> _Bm25 | None:
    with _lock:
        if index_id in _cache:
            return _cache[index_id]
    rows = db.execute(select(IndexChunk.id, IndexChunk.document_id, IndexChunk.ctype, IndexChunk.text)
                      .where(IndexChunk.index_id == index_id).order_by(IndexChunk.chunk_index)).all()
    if not rows:
        return None
    toks = [tokenize(r.text) or ["_"] for r in rows]
    entry = _Bm25(BM25Okapi(toks), [r.id for r in rows], [r.document_id for r in rows], [r.ctype for r in rows],
                  [set(t) for t in toks])
    with _lock:
        _cache[index_id] = entry
    return entry


def _bm25_search(db: Session, index_id: str, query: str, limit: int, document_ids, types):
    e = _bm25_for(db, index_id)
    q = tokenize(query)
    if e is None or not q:
        return [], q
    scores = np.asarray(e.bm25.get_scores(q))
    mask = np.ones(len(scores), dtype=bool)
    if document_ids:
        mask &= np.isin(e.doc_ids, document_ids)
    if types:
        mask &= np.isin(e.types, types)
    # A match means the chunk contains a query term. Do not gate on score > 0: in a tiny corpus BM25's
    # inverse document frequency for a term found in half the chunks is exactly 0.
    qset = set(q)
    mask &= np.fromiter((bool(qset & t) for t in e.tokens), dtype=bool, count=len(e.tokens))
    scores = np.where(mask, scores, -np.inf)
    order = [i for i in np.argsort(-scores)[:limit] if np.isfinite(scores[i])]
    return [(e.ids[i], float(scores[i]), [t for t in dict.fromkeys(q) if t in e.tokens[i]]) for i in order], q


def search(db: Session, index: Index, query: str, mode: str = "dense", top_k: int = 5, candidates: int | None = None,
           score_threshold: float | None = None, document_ids: list[str] | None = None,
           types: list[str] | None = None, rrf_k: int = 60) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if index.status != "ready":
        raise ValueError(f"Index is {index.status}; it can be searched once it is ready")
    cand = max(candidates or top_k * 4, top_k)
    timings: dict[str, float] = {}
    info: dict = {"mode": mode}

    dense: list = []
    if mode in ("dense", "hybrid"):
        t0 = time.perf_counter()
        emb = embed_texts(db, index.provider, [query])
        timings["embed_query_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        qv = emb.vectors[0]
        if len(qv) != index.dimension:
            raise DimensionMismatch(f"Query vector has {len(qv)} dimensions but index '{index.name}' stores "
                                    f"{index.dimension}-dimensional vectors ({index.provider}/{index.model}).")
        info["query_embedding"] = {"provider": index.provider, "model": index.model, "dimension": len(qv),
                                   "norm": round(float(np.linalg.norm(qv)), 6), "tokens_billed": emb.tokens_billed,
                                   "cached": emb.cache_hits > 0, "first_values": [round(float(x), 5) for x in qv[:8]]}
        t0 = time.perf_counter()
        dense = qdrant_store.search(index.collection, qv, cand, document_ids, types)
        timings["vector_search_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    kw: list = []
    if mode in ("bm25", "hybrid"):
        t0 = time.perf_counter()
        kw, qtoks = _bm25_search(db, index.id, query, cand, document_ids, types)
        timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        info["query_terms"] = list(dict.fromkeys(qtoks))

    parts: dict[str, dict] = {}
    for rank, h in enumerate(dense, 1):
        parts.setdefault(h.id, {})["dense"] = {"rank": rank, "score": round(h.score, 5)}
    for rank, (cid, s, matched) in enumerate(kw, 1):
        parts.setdefault(cid, {})["bm25"] = {"rank": rank, "score": round(s, 4), "matched_terms": matched}
    if mode == "hybrid":
        for p in parts.values():
            p["rrf"] = round(sum(1.0 / (rrf_k + p[m]["rank"]) for m in ("dense", "bm25") if m in p), 6)
        key = lambda cid: -parts[cid]["rrf"]  # noqa: E731
    elif mode == "dense":
        key = lambda cid: parts[cid]["dense"]["rank"]  # noqa: E731
    else:
        key = lambda cid: parts[cid]["bm25"]["rank"]  # noqa: E731
    ordered = sorted(parts, key=key)
    if score_threshold is not None:
        gate = "dense" if mode in ("dense", "hybrid") else "bm25"
        ordered = [c for c in ordered if gate in parts[c] and parts[c][gate]["score"] >= score_threshold]
    ordered = ordered[:top_k]

    rows = {r.id: r for r in db.scalars(select(IndexChunk).where(IndexChunk.id.in_(ordered or [""])))}
    results = []
    for rank, cid in enumerate(ordered, 1):
        r, p = rows[cid], parts[cid]
        score = p["rrf"] if mode == "hybrid" else (p["dense"]["score"] if mode == "dense" else p["bm25"]["score"])
        results.append({
            "rank": rank, "chunk_id": r.id, "score": score, "contributions": p,
            "document_id": r.document_id, "version_id": r.version_id, "filename": r.filename,
            "chunk_index": r.chunk_index, "type": r.ctype, "section": r.section, "pages": r.pages,
            "printed_pages": r.printed_pages, "start": r.start, "end": r.end, "tokens": r.tokens,
            "citation": citation(r), "text": r.text,
        })
    return {"query": query, "mode": mode, "top_k": top_k, "candidates": cand, "results": results,
            "timings_ms": timings, "info": info}


def citation(r: IndexChunk) -> str:
    pages = r.pages or []
    span = "" if not pages else (f"p.{pages[0]}" if len(pages) == 1 else f"pp.{pages[0]}-{pages[-1]}")
    printed = [p for p in (r.printed_pages or []) if p]
    out = f"{r.filename}" + (f", PDF {span}" if span else "")
    if printed:
        out += f" (printed {printed[0]}" + (f"-{printed[-1]}" if len(printed) > 1 else "") + ")"
    return out
