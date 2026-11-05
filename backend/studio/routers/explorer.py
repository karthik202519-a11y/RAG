"""Embedding explorer: look at stored vectors (2D projection) and compare similarity measures."""
import random
import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..embeddings import ProviderError, embed_texts
from ..models import IndexChunk, User
from ..security import current_user
from ..vectorstores import store_for
from .indexes import load_index

router = APIRouter(tags=["embedding-explorer"])


def _vectors(index, ids: list[str]) -> dict[str, np.ndarray]:
    return store_for(index.store).get_vectors(index.collection, ids)


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Project rows of x onto their first two principal components. Returns (coords, explained-variance ratios)."""
    c = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    var = s ** 2
    total = var.sum() or 1.0
    k = min(2, vt.shape[0])
    coords = c @ vt[:k].T
    if k < 2:
        coords = np.hstack([coords, np.zeros((len(x), 2 - k))])
    return coords, [float(v / total) for v in var[:2]]


@router.get("/indexes/{index_id}/explore")
def explore(index_id: str, sample: int = Query(300, ge=3, le=2000), seed: int = 1, document_id: str | None = None,
            user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """A random sample of the index's chunks projected to 2D with PCA."""
    i = load_index(db, user, index_id)
    if i.status != "ready":
        raise HTTPException(409, f"Index is {i.status}")
    stmt = select(IndexChunk).where(IndexChunk.index_id == i.id)
    if document_id:
        stmt = stmt.where(IndexChunk.document_id == document_id)
    rows = db.scalars(stmt.order_by(IndexChunk.id)).all()
    if len(rows) < 3:
        raise HTTPException(422, "Need at least 3 chunks to project.")
    rows = random.Random(seed).sample(rows, min(sample, len(rows)))
    vecs = _vectors(i, [r.id for r in rows])
    rows = [r for r in rows if r.id in vecs]
    mat = np.vstack([vecs[r.id] for r in rows])
    coords, ratio = pca_2d(mat)
    return {"index_id": i.id, "dimension": int(mat.shape[1]), "sampled": len(rows), "total_chunks": i.chunk_count,
            "explained_variance": [round(v, 4) for v in ratio], "method": "PCA (first two principal components)",
            "warning": "A 2D projection of a high-dimensional space distorts distances. Points that look close may be far apart, "
                       f"and these two axes keep only {sum(ratio) * 100:.1f}% of the variance.",
            "points": [{"chunk_id": r.id, "x": round(float(c[0]), 5), "y": round(float(c[1]), 5), "filename": r.filename,
                        "chunk_index": r.chunk_index, "type": r.ctype, "pages": r.pages,
                        "preview": r.text[:90].replace("\n", " ")} for r, c in zip(rows, coords)]}


class SimIn(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    chunk_ids: list[str] = Field(min_length=1, max_length=12)


@router.post("/indexes/{index_id}/similarity")
def similarity(index_id: str, body: SimIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Score chosen chunks against a query with cosine, dot product and Euclidean distance, and rank them under each."""
    i = load_index(db, user, index_id)
    if i.status != "ready":
        raise HTTPException(409, f"Index is {i.status}")
    chosen = {r.id: r for r in db.scalars(select(IndexChunk).where(IndexChunk.index_id == i.id, IndexChunk.id.in_(body.chunk_ids)))}
    missing = [c for c in body.chunk_ids if c not in chosen]
    if missing:
        raise HTTPException(404, f"Chunks not in this index: {missing}")
    try:
        q = embed_texts(db, i.provider, [body.query]).vectors[0]
    except ProviderError as e:
        raise HTTPException(502, str(e)) from None
    vecs = _vectors(i, body.chunk_ids)
    if len(q) != i.dimension:
        raise HTTPException(409, f"Query has {len(q)} dimensions but the index stores {i.dimension}.")
    rows = []
    for cid in body.chunk_ids:
        v = vecs[cid]
        cos = float(v @ q / ((np.linalg.norm(v) * np.linalg.norm(q)) or 1.0))
        rows.append({"chunk_id": cid, "filename": chosen[cid].filename, "chunk_index": chosen[cid].chunk_index,
                     "preview": chosen[cid].text[:90].replace("\n", " "), "cosine": round(cos, 5), "dot": round(float(v @ q), 5),
                     "euclidean": round(float(np.linalg.norm(v - q)), 5), "norm": round(float(np.linalg.norm(v)), 5)})
    for key, rev in (("cosine", True), ("dot", True), ("euclidean", False)):
        for rank, r in enumerate(sorted(rows, key=lambda r: r[key], reverse=rev), 1):
            r[f"rank_{key}"] = rank
    return {"query": body.query, "query_norm": round(float(np.linalg.norm(q)), 5), "dimension": int(len(q)), "rows": rows,
            "note": "For unit-length vectors cosine equals the dot product and Euclidean distance is sqrt(2 - 2*cosine), so all three "
                    "rank identically. They differ only when vector lengths differ."}


class MatrixIn(BaseModel):
    chunk_ids: list[str] = Field(min_length=2, max_length=16)


@router.post("/indexes/{index_id}/similarity-matrix")
def similarity_matrix(index_id: str, body: MatrixIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Pairwise cosine similarity between chosen chunks."""
    i = load_index(db, user, index_id)
    chosen = {r.id: r for r in db.scalars(select(IndexChunk).where(IndexChunk.index_id == i.id, IndexChunk.id.in_(body.chunk_ids)))}
    missing = [c for c in body.chunk_ids if c not in chosen]
    if missing:
        raise HTTPException(404, f"Chunks not in this index: {missing}")
    vecs = _vectors(i, body.chunk_ids)
    m = np.vstack([vecs[c] for c in body.chunk_ids])
    m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return {"chunks": [{"chunk_id": c, "label": f"{chosen[c].filename} #{chosen[c].chunk_index}",
                        "preview": chosen[c].text[:70].replace("\n", " ")} for c in body.chunk_ids],
            "matrix": [[round(float(x), 4) for x in row] for row in (m @ m.T)]}
