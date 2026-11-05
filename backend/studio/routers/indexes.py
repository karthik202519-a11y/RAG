import time
import uuid
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import generation, jobs, retrieval
from ..chunking import ChunkingError, validate_params
from ..config import settings
from ..db import get_db
from ..embeddings import PROVIDERS, ProviderError, embed_texts, get_provider
from ..models import AuditEvent, Document, DocumentVersion, Index, IndexChunk, Job, RetrievalRun, User
from ..retrieval import DimensionMismatch, pipeline
from ..security import current_user, get_project
from ..vectorstores import STORES, store_for

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
    models = generation.configured_models()
    default = generation.default_model()
    gen_ok, gen_detail = generation.status_for(default)
    stores = []
    for name, store in STORES.items():
        ok, detail = store.status()
        stores.append({"id": name, "available": ok, "mode": store.mode(), "detail": detail})
    return {"embedding": emb,
            "vector_stores": stores,
            "generation": {"configured": gen_ok, "detail": gen_detail,
                           "default_model": default or None, "models": models},
            "vision": {"configured": generation.vision_status()[0], "detail": generation.vision_status()[1],
                       "model": settings.azure_vision_deployment or None}}


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
    store: Literal["qdrant", "faiss", "chroma", "lancedb", "pgvector"] = "qdrant"
    contextual: bool = Field(False, description="A chat model writes a situating sentence per text chunk, embedded with it (one call per chunk)")
    contextual_model: str | None = Field(None, max_length=100)


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
    store_ok, store_detail = store_for(body.store).status()
    if not store_ok:
        raise HTTPException(409, f"Vector store is unavailable. {store_detail}")
    if body.contextual:
        gok, gwhy = generation.status_for(body.contextual_model or generation.default_model())
        if not gok:
            raise HTTPException(409, f"Contextual embedding needs the chat model. {gwhy}")

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

    idx = Index(project_id=project.id, name=body.name.strip(), provider=provider.id, model=provider.model, store=body.store,
                collection="pending", config={"chunking": {"method": body.chunking.method, "params": params},
                                              "version_ids": version_ids, "section_prefix": body.section_prefix,
                                              "embedding": {"provider": provider.id, "model": provider.model},
                                              **({"contextual": {"model": body.contextual_model or generation.default_model()}} if body.contextual else {})})
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
        out["stored_vectors"] = store_for(i.store).count(i.collection)
        by_doc = db.execute(select(IndexChunk.filename, func.count()).where(IndexChunk.index_id == i.id)
                            .group_by(IndexChunk.filename)).all()
        out["chunks_by_document"] = {f: n for f, n in by_doc}
    return out


@router.delete("/indexes/{index_id}", status_code=204)
def delete_index(index_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    i = load_index(db, user, index_id, "editor")
    store_for(i.store).delete_collection(i.collection)
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
                                       "tokens": r.tokens, "pages": r.pages, "section": r.section, "text": r.text, "context": r.context}
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
    got = store_for(i.store).get_vectors(i.collection, [chunk_id])
    if chunk_id not in got:
        raise HTTPException(404, "Vector not found")
    v = got[chunk_id]
    return {"chunk_id": chunk_id, "dimension": int(len(v)), "norm": round(float(np.linalg.norm(v)), 6),
            "min": round(float(v.min()), 5), "max": round(float(v.max()), 5), "first_values": [round(float(x), 5) for x in v[:n]]}


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["dense", "bm25", "hybrid", "tfidf", "graph", "graph_hybrid"] = "dense"
    top_k: int = Field(5, ge=1, le=50)
    candidates: int | None = Field(None, ge=1, le=200, description="Candidates fetched per retriever before fusion")
    score_threshold: float | None = None
    document_ids: list[str] | None = None
    types: list[Literal["text", "table"]] | None = None
    rrf_k: int = Field(60, ge=1, le=1000)
    expand: Literal["none", "parent", "neighbors"] = "none"
    neighbors: int = Field(1, ge=1, le=3)


@router.post("/indexes/{index_id}/search")
def search_index(index_id: str, body: SearchIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    i = load_index(db, user, index_id)
    try:
        out = retrieval.search(db, i, body.query, body.mode, body.top_k, body.candidates, body.score_threshold,
                               body.document_ids, body.types, body.rrf_k, body.expand, body.neighbors)
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


class RetrieveIn(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["dense", "bm25", "hybrid", "tfidf", "graph", "graph_hybrid"] = "dense"
    top_k: int = Field(5, ge=1, le=50)
    transform: Literal["none", "rewrite", "multi_query", "hyde", "decompose"] = "none"
    n_queries: int = Field(3, ge=1, le=6)
    mmr: bool = False
    mmr_lambda: float = Field(0.6, ge=0.0, le=1.0, description="1 = pure relevance, 0 = pure diversity")
    rerank: Literal["none", "llm", "cross_encoder"] = "none"
    rerank_candidates: int = Field(20, ge=2, le=50)
    llm_model: str | None = Field(None, max_length=100)
    document_ids: list[str] | None = None
    types: list[Literal["text", "table"]] | None = None
    score_threshold: float | None = None
    rrf_k: int = Field(60, ge=1, le=1000)
    expand: Literal["none", "parent", "neighbors"] = "none"
    neighbors: int = Field(1, ge=1, le=3)


@router.post("/indexes/{index_id}/retrieve")
def retrieve_pipeline(index_id: str, body: RetrieveIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Multi-stage retrieval: optional query transformation, retrieval, fusion, MMR and LLM reranking, each reported as a stage."""
    i = load_index(db, user, index_id)
    if i.status != "ready":
        raise HTTPException(409, f"Index is {i.status}; it can be searched once it is ready")
    try:
        out = pipeline.run(db, i, body.query, mode=body.mode, top_k=body.top_k, transform=body.transform, n_queries=body.n_queries,
                           mmr_on=body.mmr, mmr_lambda=body.mmr_lambda, rerank=body.rerank, rerank_candidates=body.rerank_candidates,
                           llm_model=body.llm_model, document_ids=body.document_ids, types=body.types,
                           score_threshold=body.score_threshold, rrf_k=body.rrf_k, expand=body.expand, neighbors=body.neighbors)
    except generation.GenerationNotConfigured as e:
        raise HTTPException(409, f"This option needs the chat model, which is not configured. {e}") from None
    except DimensionMismatch as e:
        raise HTTPException(409, str(e)) from None
    except (generation.GenerationError, ProviderError) as e:
        raise HTTPException(502, str(e)) from None
    except (pipeline.PipelineError, ValueError) as e:
        raise HTTPException(422, str(e)) from None
    run = RetrievalRun(index_id=i.id, user_id=user.id, query=body.query, mode=f"pipeline:{body.mode}",
                       params=body.model_dump(exclude={"query"}), result_ids=[r["chunk_id"] for r in out["results"]],
                       timings_ms=out["timings_ms"])
    db.add(run)
    db.commit()
    out["run_id"] = run.id
    return out


class AnswerIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    model: str | None = Field(None, max_length=100, description="Chat deployment; default AZURE_OPENAI_CHAT_DEPLOYMENT")
    mode: Literal["dense", "bm25", "hybrid", "tfidf", "graph", "graph_hybrid"] = "dense"
    top_k: int = Field(5, ge=1, le=20)
    transform: Literal["none", "rewrite", "multi_query", "hyde", "decompose"] = "none"
    rerank: Literal["none", "llm", "cross_encoder"] = "none"
    mmr: bool = False
    strict: bool = True
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(600, ge=16, le=8000)
    context_budget: int = Field(3000, ge=20, le=30000, description="Tokens of retrieved context allowed in the prompt")
    min_score: float | None = Field(None, description="Abstain without calling the model if the best dense score is lower")
    document_ids: list[str] | None = None
    system_prompt: str | None = Field(None, max_length=4000)
    price_in_per_million: float | None = Field(None, ge=0)
    price_out_per_million: float | None = Field(None, ge=0)
    expand: Literal["none", "parent", "neighbors"] = "none"
    neighbors: int = Field(1, ge=1, le=3)


@router.post("/indexes/{index_id}/answer")
def answer_question(index_id: str, body: AnswerIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Retrieve, build a grounded prompt, ask the chat model, and verify its citations against the passages sent."""
    i = load_index(db, user, index_id)
    try:
        retrieved = pipeline.run(db, i, body.question, mode=body.mode, top_k=body.top_k, transform=body.transform, rerank=body.rerank, mmr_on=body.mmr, llm_model=body.model, expand=body.expand, neighbors=body.neighbors) if body.transform != "none" or body.rerank != "none" or body.mmr else None
        out = generation.answer(db, i, body.question, model=body.model, mode=body.mode, top_k=body.top_k, strict=body.strict,
                                 temperature=body.temperature, max_output_tokens=body.max_output_tokens,
                                 context_budget=body.context_budget, min_score=body.min_score,
                                 price_in_per_million=body.price_in_per_million, price_out_per_million=body.price_out_per_million,
                                 system_prompt=body.system_prompt, document_ids=body.document_ids,
                                 expand=body.expand, neighbors=body.neighbors, retrieved=retrieved)
        usage = out.get("usage") or {}
        db.add(RetrievalRun(index_id=i.id, user_id=user.id, query=body.question, mode=f"answer:{body.mode}",
                            params={"model": out.get("model"), "abstained": out.get("abstained"), "top_k": body.top_k,
                                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                                    "cost_usd": usage.get("estimated_cost_usd")},
                            result_ids=[p["chunk_id"] for p in out.get("context_passages", [])],
                            timings_ms=out.get("timings_ms", {})))
        db.commit()
        return out
    except generation.GenerationNotConfigured as e:
        raise HTTPException(409, f"Answer generation is not configured. {e}") from None
    except DimensionMismatch as e:
        raise HTTPException(409, str(e)) from None
    except (generation.GenerationError, ProviderError) as e:
        raise HTTPException(502, str(e)) from None
    except ValueError as e:
        raise HTTPException(409, str(e)) from None


class JudgeIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=20000)
    passages: list[dict] = Field(min_length=1, max_length=30, description="context_passages from the answer response")
    model: str | None = Field(None, max_length=100)


@router.post("/indexes/{index_id}/judge")
def judge_answer_endpoint(index_id: str, body: JudgeIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Check an answer's claims against the passages it was given. Needs the chat deployment."""
    load_index(db, user, index_id)
    from ..generation.judge import judge_answer
    passages = [{"n": p.get("n"), "text": str(p.get("text", ""))} for p in body.passages if isinstance(p.get("n"), int)]
    if not passages:
        raise HTTPException(422, "Each passage needs an integer 'n' and 'text'")
    try:
        return judge_answer(body.question, body.answer, passages, body.model)
    except generation.GenerationNotConfigured as e:
        raise HTTPException(409, f"The judge is not configured. {e}") from None
    except generation.GenerationError as e:
        raise HTTPException(502, str(e)) from None


@router.get("/indexes/{index_id}/runs")
def list_runs(index_id: str, limit: int = Query(20, ge=1, le=200), user: User = Depends(current_user),
              db: Session = Depends(get_db)) -> list[dict]:
    i = load_index(db, user, index_id)
    rows = db.scalars(select(RetrievalRun).where(RetrievalRun.index_id == i.id).order_by(RetrievalRun.created_at.desc()).limit(limit)).all()
    return [{"id": r.id, "query": r.query, "mode": r.mode, "params": r.params, "results": len(r.result_ids),
             "timings_ms": r.timings_ms, "created_at": r.created_at.isoformat()} for r in rows]
