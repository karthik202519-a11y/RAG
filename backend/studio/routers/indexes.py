import time
import uuid
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import jobs, retrieval
from ..chunking import ChunkingError, validate_params
from ..config import settings
from ..db import get_db
from ..embeddings import PROVIDERS, ProviderError, embed_texts, get_provider
from ..models import AuditEvent, Document, DocumentVersion, Index, IndexChunk, Job, RetrievalRun, User
from ..retrieval import DimensionMismatch
from ..security import current_user, get_project
from ..vectorstores import qdrant_store

router = APIRouter(tags=["indexes"])


# ------------------------------------------------------------------ providers
@router.get("/providers")
def list_providers(user: User = Depends(current_user)) -> dict:
    """Embedding providers and vector stores, with whether each can be used right now (no secrets shown)."""
    emb = []
    for p in PROVIDERS.values():
        ok, why = p.status()
        emb.append({"id": p.id, "model": p.model or None, "available": ok, "detail": why,
                    "semantic": p.id != "local_hash"})
    return {"embedding": emb,
            "vector_stores": [{"id": "qdrant", "available": True, "mode": qdrant_store.mode()}],
            "generation": {"configured": bool(settings.azure_chat_deployment and settings.azure_endpoint and settings.azure_api_key),
                           "detail": "Set AZURE_OPENAI_CHAT_DEPLOYMENT to enable answer generation (not built yet)."}}


@router.post("/providers/{provider_id}/test")
def test_provider(provider_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    try:
        p = get_provider(provider_id)
    except ProviderError as e:
        raise HTTPException(404, str(e)) from None
    ok, why = p.status()
    if not ok:
        raise HTTPException(409, why)
    t0 = time.perf_counter()
    try:
        r = p.embed(["connection test"])
    except ProviderError as e:
        raise HTTPException(502, str(e)) from None
    return {"provider": p.id, "model": p.model, "dimension": int(r.vectors.shape[1]), "tokens": r.tokens,
            "latency_ms": round((time.perf_counter() - t0) * 1000)}


# -------------------------------------------------------------------- indexes
class ChunkingIn(BaseModel):
    method: str
    params: dict = Field(default_factory=dict)


class IndexIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    chunking: ChunkingIn
    provider: str
    document_ids: list[str] | None = Field(None, description="Default: every ready document in the project")
    section_prefix: bool = True


def index_out(db: Session, i: Index, detail: bool = False) -> dict:
    job = db.scalar(select(Job).where(Job.subject_id == i.id, Job.kind == "build_index").order_by(Job.created_at.desc()))
    out = {"id": i.id, "project_id": i.project_id, "name": i.name, "status": i.status, "error": i.error,
           "provider": i.provider, "model": i.model, "dimension": i.dimension, "metric": i.metric, "store": i.store,
           "collection": i.collection, "chunk_count": i.chunk_count, "tokens_embedded": i.tokens_embedded,
           "tokens_billed": i.tokens_billed, "cache_hits": i.cache_hits, "build_seconds": i.build_seconds,
           "created_at": i.created_at.isoformat(), "chunking": i.config["chunking"],
           "documents": len(i.config["version_ids"]), "section_prefix": i.config["section_prefix"],
           "job": None if job is None else {"id": job.id, "status": job.status, "progress": round(job.progress, 3),
                                            "stage": job.stage, "attempts": job.attempts, "error": job.error}}
    if detail:
        out["version_ids"] = i.config["version_ids"]
    return out


def load_index(db: Session, user: User, index_id: str, role: str = "viewer") -> Index:
    i = db.get(Index, index_id)
    if i is None:
        raise HTTPException(404, "Index not found")
    get_project(db, user, i.project_id, role)
    return i


@router.post("/projects/{project_id}/indexes", status_code=201)
def create_index(project_id: str, body: IndexIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    project = get_project(db, user, project_id, "editor")
    try:
        provider = get_provider(body.provider)
    except ProviderError as e:
        raise HTTPException(422, str(e)) from None
    ok, why = provider.status()
    if not ok:
        raise HTTPException(409, f"Embedding provider '{provider.id}' is not configured. {why}")
    try:
        params = validate_params(body.chunking.method, body.chunking.params)
    except ChunkingError as e:
        raise HTTPException(422, str(e)) from None

    stmt = select(Document).where(Document.project_id == project.id)
    if body.document_ids:
        stmt = stmt.where(Document.id.in_(body.document_ids))
    version_ids = []
    for d in db.scalars(stmt).all():
        latest = d.versions[-1] if d.versions else None
        if latest is not None and latest.status == "ready":
            version_ids.append(latest.id)
    if not version_ids:
        raise HTTPException(422, "No ready documents to index. Upload and parse a document first.")

    idx = Index(project_id=project.id, name=body.name.strip(), provider=provider.id, model=provider.model,
                collection="pending", config={"chunking": {"method": body.chunking.method, "params": params},
                                              "version_ids": version_ids, "section_prefix": body.section_prefix,
                                              "embedding": {"provider": provider.id, "model": provider.model}})
    db.add(idx)
    db.flush()
    idx.collection = f"idx_{idx.id}"  # a unique namespace per index: nothing is ever overwritten
    db.add(AuditEvent(user_id=user.id, workspace_id=project.workspace_id, action="index.create", target=idx.id))
    db.commit()
    jobs.enqueue(db, project.id, "build_index", idx.id)
    return index_out(db, idx)


@router.get("/projects/{project_id}/indexes")
def list_indexes(project_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    get_project(db, user, project_id)
    rows = db.scalars(select(Index).where(Index.project_id == project_id).order_by(Index.created_at.desc())).all()
    return [index_out(db, i) for i in rows]


@router.get("/indexes/{index_id}")
def read_index(index_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    i = load_index(db, user, index_id)
    out = index_out(db, i, detail=True)
    if i.status == "ready":
        out["stored_vectors"] = qdrant_store.count(i.collection)
        by_doc = db.execute(select(IndexChunk.filename, func.count()).where(IndexChunk.index_id == i.id)
                            .group_by(IndexChunk.filename)).all()
        out["chunks_by_document"] = {f: n for f, n in by_doc}
    return out


@router.delete("/indexes/{index_id}", status_code=204)
def delete_index(index_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    i = load_index(db, user, index_id, "editor")
    qdrant_store.delete_collection(i.collection)
    retrieval.drop_bm25(i.id)
    db.query(Job).filter(Job.subject_id == i.id).delete()
    db.add(AuditEvent(user_id=user.id, action="index.delete", target=i.id, detail={"name": i.name}))
    db.delete(i)
    db.commit()


@router.get("/indexes/{index_id}/chunks")
def browse_chunks(index_id: str, limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0),
                  document_id: str | None = None, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """The stored records, for inspection."""
    i = load_index(db, user, index_id)
    stmt = select(IndexChunk).where(IndexChunk.index_id == i.id)
    if document_id:
        stmt = stmt.where(IndexChunk.document_id == document_id)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(stmt.order_by(IndexChunk.filename, IndexChunk.chunk_index).offset(offset).limit(limit)).all()
    return {"total": total, "items": [{"id": r.id, "filename": r.filename, "chunk_index": r.chunk_index, "type": r.ctype,
                                       "tokens": r.tokens, "pages": r.pages, "section": r.section, "text": r.text}
                                      for r in rows]}


@router.get("/indexes/{index_id}/chunks/{chunk_id}/vector")
def chunk_vector(index_id: str, chunk_id: str, n: int = Query(16, ge=1, le=64), user: User = Depends(current_user),
                 db: Session = Depends(get_db)) -> dict:
    """The stored vector of one chunk: dimension, norm and its first values."""
    i = load_index(db, user, index_id)
    if i.status != "ready":
        raise HTTPException(409, f"Index is {i.status}")
    row = db.get(IndexChunk, chunk_id)
    if row is None or row.index_id != i.id:
        raise HTTPException(404, "Chunk not found")
    pts = qdrant_store.client().retrieve(i.collection, ids=[str(uuid.UUID(chunk_id))], with_vectors=True)
    if not pts:
        raise HTTPException(404, "Vector not found")
    v = np.asarray(pts[0].vector, dtype=np.float32)
    return {"chunk_id": chunk_id, "dimension": int(len(v)), "norm": round(float(np.linalg.norm(v)), 6),
            "min": round(float(v.min()), 5), "max": round(float(v.max()), 5), "first_values": [round(float(x), 5) for x in v[:n]]}


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = Field(5, ge=1, le=50)
    candidates: int | None = Field(None, ge=1, le=200, description="Candidates fetched per retriever before fusion")
    score_threshold: float | None = None
    document_ids: list[str] | None = None
    types: list[Literal["text", "table"]] | None = None
    rrf_k: int = Field(60, ge=1, le=1000)


@router.post("/indexes/{index_id}/search")
def search_index(index_id: str, body: SearchIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    i = load_index(db, user, index_id)
    try:
        out = retrieval.search(db, i, body.query, body.mode, body.top_k, body.candidates, body.score_threshold,
                               body.document_ids, body.types, body.rrf_k)
    except DimensionMismatch as e:
        raise HTTPException(409, str(e)) from None
    except ProviderError as e:
        raise HTTPException(502, str(e)) from None
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
    run = RetrievalRun(index_id=i.id, user_id=user.id, query=body.query, mode=body.mode,
                       params=body.model_dump(exclude={"query"}), result_ids=[r["chunk_id"] for r in out["results"]],
                       timings_ms=out["timings_ms"])
    db.add(run)
    db.commit()
    out["run_id"] = run.id
    return out


@router.get("/indexes/{index_id}/runs")
def list_runs(index_id: str, limit: int = Query(20, ge=1, le=200), user: User = Depends(current_user),
              db: Session = Depends(get_db)) -> list[dict]:
    i = load_index(db, user, index_id)
    rows = db.scalars(select(RetrievalRun).where(RetrievalRun.index_id == i.id).order_by(RetrievalRun.created_at.desc()).limit(limit)).all()
    return [{"id": r.id, "query": r.query, "mode": r.mode, "params": r.params, "results": len(r.result_ids),
             "timings_ms": r.timings_ms, "created_at": r.created_at.isoformat()} for r in rows]
