"""Typed, immutable recipes and observable executions using the existing pipeline engines."""
import json
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import PipelineRecipe, WorkbenchRun, Document, DocumentVersion, Index, EvalQuestion, Job
from ..security import current_user, get_project
from .. import jobs, generation
from ..embeddings import get_provider, ProviderError
from ..vectorstores import store_for
from ..chunking import validate_params, ChunkingError
from .indexes import IndexIn, RetrieveIn, load_index
from .evaluation import load_dataset

router = APIRouter(tags=["workbench"])


class RecipeIn(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    index: IndexIn
    reuse_index_id: str | None = None
    retrieval: RetrieveIn
    with_answer: bool = False
    answer_model: str | None = None


def recipe_out(row):
    return {"id": row.id, "project_id": row.project_id, "name": row.name, "version": row.version, "config": row.config, "created_at": row.created_at.isoformat()}


@router.get("/projects/{project_id}/pipelines")
def list_recipes(project_id: str, user=Depends(current_user), db: Session=Depends(get_db)):
    get_project(db, user, project_id)
    return [recipe_out(r) for r in db.scalars(select(PipelineRecipe).where(PipelineRecipe.project_id == project_id).order_by(PipelineRecipe.created_at.desc()))]


@router.post("/projects/{project_id}/pipelines", status_code=201)
def save_recipe(project_id: str, body: RecipeIn, user=Depends(current_user), db: Session=Depends(get_db)):
    get_project(db, user, project_id, "editor")
    if not body.name.strip():
        raise HTTPException(422, "Give the pipeline a name.")
    if body.reuse_index_id:
        idx = load_index(db, user, body.reuse_index_id)
        if idx.project_id != project_id or idx.status != "ready":
            raise HTTPException(422, "Reuse a ready index from this project.")
        frozen = dict(idx.config, provider=idx.provider, model=idx.model, store=idx.store)
    else:
        try:
            provider = get_provider(body.index.provider)
            params = validate_params(body.index.chunking.method, body.index.chunking.params)
        except (ProviderError, ChunkingError) as e:
            raise HTTPException(422, str(e)) from None
        for component in (provider, store_for(body.index.store)):
            ok, why = component.status()
            if not ok:
                raise HTTPException(409, why)
        if body.index.chunking.method == "semantic":
            semantic_provider = get_provider(params["provider"])
            ok, why = semantic_provider.status()
            if not ok:
                raise HTTPException(409, f"Semantic chunking provider is unavailable: {why}")
        docs = db.scalars(select(Document).where(Document.project_id == project_id)).all()
        selected = body.index.document_ids
        versions = [d.versions[-1].id for d in docs if d.versions and d.versions[-1].status == "ready" and (selected is None or d.id in selected)]
        if not versions:
            raise HTTPException(422, "Select at least one ready document.")
        if selected is not None and set(selected) != {d.id for d in docs if d.versions and d.versions[-1].id in versions}:
            raise HTTPException(422, "Every selected document must be ready and belong to this project.")
        frozen = {"chunking": {"method": body.index.chunking.method, "params": params}, "version_ids": versions,
                  "section_prefix": body.index.section_prefix, "provider": provider.id, "model": provider.model, "store": body.index.store,
                  "embedding": {"provider": provider.id, "model": provider.model}}
        if body.index.contextual:
            frozen["contextual"] = {"model": body.index.contextual_model or generation.default_model()}
    retrieval = body.retrieval.model_dump()
    # A parent's type is part of the recipe, not a decorative canvas connection.
    if retrieval["expand"] == "parent" and frozen["chunking"]["method"] != "parent_child":
        raise HTTPException(422, "Parent expansion requires a parent-child chunker.")
    needs_model = retrieval["mode"] in ("graph", "graph_hybrid") or body.with_answer or retrieval["transform"] != "none" or retrieval["rerank"] == "llm" or frozen["chunking"]["method"] in ("contextual", "proposition") or "contextual" in frozen
    if needs_model:
        model = body.answer_model or retrieval.get("llm_model") or generation.default_model()
        ok, why = generation.status_for(model)
        if not ok:
            raise HTTPException(409, why)
    config = {"index": frozen, "reuse_index_id": body.reuse_index_id, "retrieval": retrieval,
              "with_answer": body.with_answer, "answer_model": body.answer_model or generation.default_model()}
    version = (db.scalar(select(func.max(PipelineRecipe.version)).where(PipelineRecipe.project_id == project_id, PipelineRecipe.name == body.name.strip())) or 0) + 1
    row = PipelineRecipe(project_id=project_id, name=body.name.strip(), version=version, config=config)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Another version was saved at the same time. Save again.") from None
    return recipe_out(row)


def load_recipe(db, user, recipe_id):
    row = db.get(PipelineRecipe, recipe_id)
    if row is None:
        raise HTTPException(404, "Pipeline not found")
    get_project(db, user, row.project_id)
    return row


def run_out(db, row, detail=True):
    job = db.scalar(select(Job).where(Job.subject_id == row.id, Job.kind == "workbench").order_by(Job.created_at.desc()))
    return {"id": row.id, "name": row.name, "kind": row.kind, "project_id": row.project_id, "status": row.status, "error": row.error,
            "config": row.config, "artifacts": row.artifacts, "result": row.result if detail else None, "created_at": row.created_at.isoformat(),
            "job": {"id": job.id, "status": job.status, "progress": job.progress, "stage": job.stage} if job else None}


@router.post("/pipelines/{recipe_id}/run", status_code=202)
def execute(recipe_id: str, user=Depends(current_user), db: Session=Depends(get_db)):
    recipe = load_recipe(db, user, recipe_id)
    get_project(db, user, recipe.project_id, "editor")
    row = WorkbenchRun(project_id=recipe.project_id, kind="pipeline", name=f"{recipe.name} v{recipe.version}", config={"recipes": [recipe_out(recipe)]})
    db.add(row); db.commit()
    jobs.enqueue(db, recipe.project_id, "workbench", row.id)
    return run_out(db, row)


class ExperimentIn(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    pipeline_ids: list[str] = Field(min_length=2, max_length=8)
    dataset_id: str
    split: Literal["all", "train", "validation", "test"] = "all"
    with_answers: bool = False


@router.post("/projects/{project_id}/experiments", status_code=202)
def experiment(project_id: str, body: ExperimentIn, user=Depends(current_user), db: Session=Depends(get_db)):
    from ..models import EvalDataset
    get_project(db, user, project_id, "editor")
    ds = load_dataset(db, user, body.dataset_id)
    recipes = [load_recipe(db, user, i) for i in dict.fromkeys(body.pipeline_ids)]
    if ds.project_id != project_id or any(r.project_id != project_id for r in recipes):
        raise HTTPException(422, "Use pipelines and a dataset from this project.")
    if len(recipes) < 2:
        raise HTTPException(422, "Select at least two distinct pipeline versions.")
    questions = db.scalars(select(EvalQuestion).where(EvalQuestion.dataset_id == ds.id, EvalQuestion.approved.is_(True))).all()
    questions = [q for q in questions if body.split == "all" or q.split == body.split]
    if not questions:
        raise HTTPException(422, "This split has no approved questions.")
    if body.with_answers:
        for recipe in recipes:
            ok, why = generation.status_for(recipe.config["answer_model"])
            if not ok:
                raise HTTPException(409, why)
    snapshot = EvalDataset(project_id=project_id, name=f"Snapshot: {body.name}"[:200], description=f"Frozen approved questions from {ds.name}; created by Experiment Studio.")
    db.add(snapshot); db.flush()
    for q in questions:
        db.add(EvalQuestion(dataset_id=snapshot.id, position=q.position, question=q.question, qtype=q.qtype, answerable=q.answerable,
                           evidence=q.evidence, keywords=q.keywords, reference_answer=q.reference_answer, split=q.split, approved=True))
    row = WorkbenchRun(project_id=project_id, kind="experiment", name=body.name.strip(), config={"recipes": [recipe_out(r) for r in recipes],
        "dataset_id": snapshot.id, "source_dataset_id": ds.id, "questions": len(questions), "split": body.split, "with_answers": body.with_answers})
    db.add(row); db.commit()
    jobs.enqueue(db, project_id, "workbench", row.id)
    return run_out(db, row)


@router.get("/projects/{project_id}/workbench-runs")
def list_runs(project_id: str, user=Depends(current_user), db: Session=Depends(get_db)):
    get_project(db, user, project_id)
    return [run_out(db, r, False) for r in db.scalars(select(WorkbenchRun).where(WorkbenchRun.project_id == project_id).order_by(WorkbenchRun.created_at.desc()).limit(100))]


@router.get("/workbench-runs/{run_id}")
def read_run(run_id: str, user=Depends(current_user), db: Session=Depends(get_db)):
    row = db.get(WorkbenchRun, run_id)
    if row is None:
        raise HTTPException(404, "Run not found")
    get_project(db, user, row.project_id)
    return run_out(db, row)


@router.get("/workbench-runs/{run_id}/export")
def export_run(run_id: str, user=Depends(current_user), db: Session=Depends(get_db)):
    out = read_run(run_id, user, db)
    return Response(json.dumps(out, indent=2), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="workbench-{run_id}.json"'})


@router.get("/indexes/{index_id}/construction")
def construction(index_id: str, user=Depends(current_user), db: Session=Depends(get_db)):
    from ..models import IndexChunk, GraphBuild, GraphEntity, GraphEdge
    from ..jobs.index_job import embed_input
    idx = load_index(db, user, index_id)
    chunks = db.scalars(select(IndexChunk).where(IndexChunk.index_id == idx.id).order_by(IndexChunk.document_id, IndexChunk.chunk_index).limit(5)).all()
    vectors = store_for(idx.store).get_vectors(idx.collection, [c.id for c in chunks]) if idx.status == "ready" and chunks else {}
    samples = [{"id": c.id, "filename": c.filename, "pages": c.pages, "type": c.ctype, "text": c.text[:1400], "context": c.context,
                "embedding_input": embed_input(c.text, c.ctype, c.section, idx.config.get("section_prefix", True), c.context)[:1800],
                "vector_preview": [round(float(v), 6) for v in vectors[c.id][:12]] if c.id in vectors else [], "dimension": len(vectors[c.id]) if c.id in vectors else None} for c in chunks]
    edges = db.scalars(select(GraphEdge).where(GraphEdge.index_id == idx.id).limit(8)).all()
    entity_ids = {e.src_id for e in edges} | {e.dst_id for e in edges}
    entities = {e.id: e.name for e in db.scalars(select(GraphEntity).where(GraphEntity.id.in_(entity_ids or [""])))}
    evidence = {c.id: c for c in db.scalars(select(IndexChunk).where(IndexChunk.id.in_([e.chunk_id for e in edges] or [""])))}
    build = db.get(GraphBuild, idx.id)
    return {"index_id": idx.id, "chunking": idx.config["chunking"], "provider": idx.provider, "model": idx.model, "store": idx.store,
            "dimension": idx.dimension, "chunk_count": idx.chunk_count, "cache_hits": idx.cache_hits, "tokens_billed": idx.tokens_billed,
            "section_prefix": idx.config.get("section_prefix", True), "samples": samples,
            "graph": {"status": build.status if build else "not_built", "model": build.model if build else None,
                "chunks_processed": build.chunks_processed if build else 0, "relations_dropped": build.relations_dropped if build else 0,
                "edges": [{"source": entities[e.src_id], "relation": e.relation, "target": entities[e.dst_id], "chunk_id": e.chunk_id,
                           "filename": evidence[e.chunk_id].filename if e.chunk_id in evidence else None,
                           "text": evidence[e.chunk_id].text[:1600] if e.chunk_id in evidence else None} for e in edges],
                "grounding": "Entity names are checked against source text. The relation is a model proposal, not a verified fact."}}
