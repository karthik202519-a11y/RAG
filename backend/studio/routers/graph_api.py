from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .. import generation, jobs
from ..config import settings
from ..db import get_db
from ..graph import graph_overview, graph_search
from ..models import GraphBuild, GraphEdge, GraphEntity, Job, User
from ..security import current_user
from .indexes import load_index

router = APIRouter(tags=["graph"])


class BuildIn(BaseModel):
    max_chunks: int = Field(150, ge=5, le=2000, description="Chunks to read, spread evenly over the index. Each costs one chat call.")
    model: str | None = Field(None, max_length=100)
    min_tokens: int = Field(25, ge=1, le=500, description="Skip chunks shorter than this; headings and fragments rarely hold a relation")


def _build_out(b: GraphBuild | None, job: Job | None = None) -> dict | None:
    if b is None:
        return None
    return {"status": b.status, "model": b.model, "chunks_requested": b.chunks_requested, "chunks_processed": b.chunks_processed,
            "chunks_failed": b.chunks_failed, "relations_dropped": b.relations_dropped, "tokens": b.tokens, "error": b.error,
            "seconds": b.seconds, "progress": round(job.progress, 3) if job else None, "stage": job.stage if job else None}


@router.post("/indexes/{index_id}/graph", status_code=202)
def build_graph(index_id: str, body: BuildIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Extract entities and relations from the index's chunks with the chat model. Spends tokens: one call per chunk."""
    idx = load_index(db, user, index_id, "editor")
    if idx.status != "ready":
        raise HTTPException(409, f"Index is {idx.status}")
    ok, why = generation.generation_status()
    if not ok:
        raise HTTPException(409, f"Graph extraction needs the chat model. {why}")
    existing = db.get(GraphBuild, idx.id)
    if existing is not None and existing.status in ("queued", "running"):
        raise HTTPException(409, "A graph build is already running for this index")
    model = body.model or settings.azure_chat_deployment
    if existing is None:
        existing = GraphBuild(index_id=idx.id)
        db.add(existing)
    existing.status, existing.model, existing.error = "queued", model, None
    db.commit()
    job = jobs.enqueue(db, idx.project_id, "build_graph", idx.id, {"max_chunks": body.max_chunks, "min_tokens": body.min_tokens})
    return {"build": _build_out(existing, job)}


@router.get("/indexes/{index_id}/graph")
def read_graph(index_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    idx = load_index(db, user, index_id)
    b = db.get(GraphBuild, idx.id)
    job = db.scalar(select(Job).where(Job.subject_id == idx.id, Job.kind == "build_graph").order_by(Job.created_at.desc()))
    out = {"build": _build_out(b, job), "graph": None}
    if b is not None and b.status == "ready":
        out["graph"] = graph_overview(db, idx.id)
    return out


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(5, ge=1, le=20)
    hops: int = Field(1, ge=1, le=2)


@router.post("/indexes/{index_id}/graph/search")
def search_graph(index_id: str, body: SearchIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Local graph search: start at the entities the question names, follow relations, return the chunks they came from."""
    idx = load_index(db, user, index_id)
    b = db.get(GraphBuild, idx.id)
    if b is None or b.status != "ready":
        raise HTTPException(409, "Build the graph for this index first")
    return graph_search(db, idx.id, body.query, body.top_k, body.hops)


@router.delete("/indexes/{index_id}/graph", status_code=204)
def delete_graph(index_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    idx = load_index(db, user, index_id, "editor")
    db.execute(delete(GraphEdge).where(GraphEdge.index_id == idx.id))
    db.execute(delete(GraphEntity).where(GraphEntity.index_id == idx.id))
    b = db.get(GraphBuild, idx.id)
    if b is not None:
        db.delete(b)
    db.commit()
