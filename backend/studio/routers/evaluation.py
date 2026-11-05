import csv
import io
import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import jobs
from ..db import get_db
from ..evaluation import service as ev
from ..models import AuditEvent, EvalDataset, EvalQuestion, EvalRun, Index, Job, User
from ..security import current_user, get_project

router = APIRouter(tags=["evaluation"])


def load_dataset(db: Session, user: User, dataset_id: str, role: str = "viewer") -> EvalDataset:
    d = db.get(EvalDataset, dataset_id)
    if d is None:
        raise HTTPException(404, "Dataset not found")
    get_project(db, user, d.project_id, role)
    return d


def q_out(q: EvalQuestion) -> dict:
    return {"id": q.id, "position": q.position, "question": q.question, "qtype": q.qtype, "answerable": q.answerable,
            "evidence": q.evidence, "keywords": q.keywords, "reference_answer": q.reference_answer, "split": q.split,
            "approved": q.approved}


class DatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field("", max_length=2000)


@router.post("/projects/{project_id}/eval-datasets", status_code=201)
def create_dataset(project_id: str, body: DatasetIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    p = get_project(db, user, project_id, "editor")
    d = EvalDataset(project_id=p.id, name=body.name.strip(), description=body.description)
    db.add(d)
    db.add(AuditEvent(user_id=user.id, workspace_id=p.workspace_id, action="eval_dataset.create", target=d.id))
    db.commit()
    return {"id": d.id, "name": d.name, "description": d.description, "questions": 0}


@router.get("/projects/{project_id}/eval-datasets")
def list_datasets(project_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    get_project(db, user, project_id)
    out = []
    for d in db.scalars(select(EvalDataset).where(EvalDataset.project_id == project_id).order_by(EvalDataset.created_at.desc())):
        n = db.scalar(select(func.count()).select_from(EvalQuestion).where(EvalQuestion.dataset_id == d.id)) or 0
        runs = db.scalar(select(func.count()).select_from(EvalRun).where(EvalRun.dataset_id == d.id)) or 0
        out.append({"id": d.id, "name": d.name, "description": d.description, "questions": n, "runs": runs,
                    "created_at": d.created_at.isoformat()})
    return out


@router.get("/eval-datasets/{dataset_id}")
def read_dataset(dataset_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    d = load_dataset(db, user, dataset_id)
    qs = db.scalars(select(EvalQuestion).where(EvalQuestion.dataset_id == d.id).order_by(EvalQuestion.position)).all()
    splits: dict[str, int] = {}
    for q in qs:
        splits[q.split] = splits.get(q.split, 0) + 1
    test_runs = db.scalar(select(func.count()).select_from(EvalRun).where(EvalRun.dataset_id == d.id,
                                                                          EvalRun.config["split"].as_string() == "test")) or 0
    return {"id": d.id, "project_id": d.project_id, "name": d.name, "description": d.description, "questions": [q_out(q) for q in qs],
            "splits": splits, "answerable": sum(q.answerable for q in qs), "unanswerable": sum(not q.answerable for q in qs),
            "runs_on_test_split": test_runs,
            "guidance": "Tune settings on the train/validation splits and use the test split once for the final claim. "
                        "Repeatedly choosing the best configuration on the test split overstates how well it will do on new questions."}


@router.delete("/eval-datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    d = load_dataset(db, user, dataset_id, "editor")
    db.delete(d)
    db.commit()


class EvidenceIn(BaseModel):
    document: str = Field("", max_length=300, description="Document id or a fragment of its filename; empty = any document")
    pages: list[int] = Field(default_factory=list)


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    qtype: str = Field("", max_length=40)
    answerable: bool = True
    evidence: list[EvidenceIn] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    reference_answer: str | None = Field(None, max_length=8000)
    split: Literal["train", "validation", "test"] = "test"
    approved: bool = True


@router.post("/eval-datasets/{dataset_id}/questions", status_code=201)
def add_question(dataset_id: str, body: QuestionIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    d = load_dataset(db, user, dataset_id, "editor")
    if body.answerable and not body.evidence:
        raise HTTPException(422, "An answerable question needs at least one evidence item (document and pages), "
                                 "or mark it unanswerable.")
    pos = db.scalar(select(func.count()).select_from(EvalQuestion).where(EvalQuestion.dataset_id == d.id)) or 0
    q = EvalQuestion(dataset_id=d.id, position=pos, question=body.question.strip(), qtype=body.qtype, answerable=body.answerable,
                     evidence=[e.model_dump() for e in body.evidence] if body.answerable else [], keywords=body.keywords,
                     reference_answer=body.reference_answer, split=body.split, approved=body.approved)
    db.add(q)
    db.commit()
    return q_out(q)


@router.patch("/eval-questions/{question_id}")
def patch_question(question_id: str, body: QuestionIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    q = db.get(EvalQuestion, question_id)
    if q is None:
        raise HTTPException(404, "Question not found")
    load_dataset(db, user, q.dataset_id, "editor")
    if body.answerable and not body.evidence:
        raise HTTPException(422, "An answerable question needs at least one evidence item.")
    q.question, q.qtype, q.answerable = body.question.strip(), body.qtype, body.answerable
    q.evidence = [e.model_dump() for e in body.evidence] if body.answerable else []
    q.keywords, q.reference_answer, q.split, q.approved = body.keywords, body.reference_answer, body.split, body.approved
    db.commit()
    return q_out(q)


@router.delete("/eval-questions/{question_id}", status_code=204)
def delete_question(question_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    q = db.get(EvalQuestion, question_id)
    if q is None:
        raise HTTPException(404, "Question not found")
    load_dataset(db, user, q.dataset_id, "editor")
    db.delete(q)
    db.commit()


class ImportIn(BaseModel):
    questions: list[dict] = Field(min_length=1, max_length=5000)
    format: Literal["auto", "native", "prototype_basic", "prototype_hard"] = "auto"
    default_document: str = Field("", max_length=300, description="Document fragment for formats that do not name one")
    skip_missing_documents: bool = True


@router.post("/eval-datasets/{dataset_id}/import")
def import_questions(dataset_id: str, body: ImportIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    d = load_dataset(db, user, dataset_id, "editor")
    try:
        return ev.import_questions(db, d.id, d.project_id, body.questions, body.format, body.default_document, body.skip_missing_documents)
    except ev.ImportError_ as e:
        raise HTTPException(422, str(e)) from None


class SplitIn(BaseModel):
    train: float = Field(0.6, ge=0, le=1)
    validation: float = Field(0.2, ge=0, le=1)
    seed: int = 1


@router.post("/eval-datasets/{dataset_id}/assign-splits")
def assign_splits(dataset_id: str, body: SplitIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    d = load_dataset(db, user, dataset_id, "editor")
    if body.train + body.validation > 1:
        raise HTTPException(422, "train + validation must not exceed 1; the remainder becomes the test split.")
    return ev.assign_splits(db, d.id, body.train, body.validation, body.seed)


# ------------------------------------------------------------------------- runs
class RunIn(BaseModel):
    index_id: str
    label: str = Field("", max_length=200)
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    k: int = Field(5, ge=1, le=50)
    split: Literal["all", "train", "validation", "test"] = "all"
    transform: Literal["none", "rewrite", "multi_query", "hyde", "decompose"] = "none"
    rerank: Literal["none", "llm", "cross_encoder"] = "none"
    mmr: bool = False
    with_answers: bool = False
    judge: bool = Field(False, description="Also judge faithfulness of each answer with the chat model (extra LLM calls)")
    llm_model: str | None = Field(None, max_length=100)


def run_out(r: EvalRun, detail: bool = False, job: Job | None = None) -> dict:
    out = {"id": r.id, "dataset_id": r.dataset_id, "index_id": r.index_id, "index_name": r.index_name, "label": r.label,
           "config": r.config, "status": r.status, "error": r.error, "metrics": r.metrics, "seconds": r.seconds,
           "created_at": r.created_at.isoformat(),
           "job": None if job is None else {"progress": round(job.progress, 3), "stage": job.stage, "status": job.status}}
    if detail:
        out["per_question"] = r.per_question
    return out


@router.post("/eval-datasets/{dataset_id}/runs", status_code=201)
def start_run(dataset_id: str, body: RunIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    d = load_dataset(db, user, dataset_id, "editor")
    idx = db.get(Index, body.index_id)
    if idx is None or idx.project_id != d.project_id:
        raise HTTPException(404, "Index not found in this project")
    if idx.status != "ready":
        raise HTTPException(409, f"Index is {idx.status}")
    stmt = select(func.count()).select_from(EvalQuestion).where(EvalQuestion.dataset_id == d.id, EvalQuestion.approved.is_(True))
    if body.split != "all":
        stmt = stmt.where(EvalQuestion.split == body.split)
    if not (db.scalar(stmt) or 0):
        raise HTTPException(422, "No approved questions in that split.")
    r = EvalRun(dataset_id=d.id, index_id=idx.id, index_name=idx.name, label=body.label or f"{idx.name} · {body.mode} k={body.k}",
                config=body.model_dump(exclude={"index_id", "label"}))
    db.add(r)
    db.commit()
    job = jobs.enqueue(db, d.project_id, "run_eval", r.id)
    return run_out(r, job=job)


MAX_GRID = 24


class GridIn(BaseModel):
    index_ids: list[str] = Field(min_length=1, max_length=8)
    modes: list[Literal["dense", "bm25", "hybrid"]] = Field(default_factory=lambda: ["dense"], min_length=1, max_length=3)
    ks: list[int] = Field(default_factory=lambda: [5], min_length=1, max_length=4)
    rerank: list[Literal["none", "llm", "cross_encoder"]] = Field(default_factory=lambda: ["none"], min_length=1, max_length=3)
    split: Literal["all", "train", "validation", "test"] = "all"
    transform: Literal["none", "rewrite", "multi_query", "hyde", "decompose"] = "none"
    mmr: bool = False


@router.post("/eval-datasets/{dataset_id}/grid", status_code=201)
def start_grid(dataset_id: str, body: GridIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Run every combination of index x mode x k x rerank on the same questions, so results compare like with like."""
    combos = [(i, m, k, rr) for i in dict.fromkeys(body.index_ids) for m in dict.fromkeys(body.modes)
              for k in dict.fromkeys(body.ks) for rr in dict.fromkeys(body.rerank)]
    if len(combos) > MAX_GRID:
        raise HTTPException(422, f"That grid is {len(combos)} runs; the limit is {MAX_GRID}. Narrow one of the lists.")
    runs = []
    for i, m, k, rr in combos:
        runs.append(start_run(dataset_id, RunIn(index_id=i, mode=m, k=k, rerank=rr, split=body.split, transform=body.transform, mmr=body.mmr),
                              user, db))
    return {"count": len(runs), "runs": runs}


@router.get("/eval-datasets/{dataset_id}/runs")
def list_runs(dataset_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    d = load_dataset(db, user, dataset_id)
    rows = db.scalars(select(EvalRun).where(EvalRun.dataset_id == d.id).order_by(EvalRun.created_at.desc())).all()
    jobs_by = {j.subject_id: j for j in db.scalars(select(Job).where(Job.kind == "run_eval", Job.subject_id.in_([r.id for r in rows] or [""])))}
    return [run_out(r, job=jobs_by.get(r.id)) for r in rows]


def load_run(db: Session, user: User, run_id: str) -> EvalRun:
    r = db.get(EvalRun, run_id)
    if r is None:
        raise HTTPException(404, "Run not found")
    load_dataset(db, user, r.dataset_id)
    return r


@router.get("/eval-runs/{run_id}")
def read_run(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    r = load_run(db, user, run_id)
    job = db.scalar(select(Job).where(Job.kind == "run_eval", Job.subject_id == r.id))
    return run_out(r, detail=True, job=job)


@router.delete("/eval-runs/{run_id}", status_code=204)
def delete_run(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    r = load_run(db, user, run_id)
    load_dataset(db, user, r.dataset_id, "editor")
    db.query(Job).filter(Job.kind == "run_eval", Job.subject_id == r.id).delete()
    db.delete(r)
    db.commit()


@router.get("/eval-datasets/{dataset_id}/compare")
def compare(dataset_id: str, runs: str = Query(..., description="Comma-separated run ids; the first is the baseline"),
            user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    d = load_dataset(db, user, dataset_id)
    ids = [x for x in runs.split(",") if x]
    rows = {r.id: r for r in db.scalars(select(EvalRun).where(EvalRun.id.in_(ids), EvalRun.dataset_id == d.id))}
    if len(rows) != len(set(ids)):
        raise HTTPException(404, "One or more runs were not found in this dataset")
    chosen = [rows[i] for i in ids]
    if any(r.status != "done" for r in chosen):
        raise HTTPException(409, "All runs must be finished to compare them")
    try:
        return ev.compare_runs(chosen)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.get("/eval-runs/{run_id}/export")
def export_run(run_id: str, format: Literal["json", "csv"] = "json", user: User = Depends(current_user), db: Session = Depends(get_db)) -> Response:
    r = load_run(db, user, run_id)
    if r.status != "done":
        raise HTTPException(409, "The run is not finished")
    if format == "json":
        return Response(json.dumps(run_out(r, detail=True), indent=1), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="eval-{r.id}.json"'})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["question", "type", "answerable", "split", "first_rank", "hit", "recall", "precision", "mrr", "ndcg", "latency_ms", "top1_dense_score"])
    for p in r.per_question or []:
        m = p["metrics"]
        w.writerow([p["question"], p["qtype"], p["answerable"], p["split"], m.get("first_rank"), m.get("hit"), m.get("recall"),
                    m.get("precision"), m.get("mrr"), m.get("ndcg"), p["latency_ms"], p["top1_dense_score"]])
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="eval-{r.id}.csv"'})
