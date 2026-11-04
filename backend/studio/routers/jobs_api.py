import asyncio
import json

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import db as dbmod
from .. import jobs
from ..config import settings
from ..db import get_db
from ..models import Document, DocumentVersion, Job, Project, User
from ..security import current_user, get_project, workspace_role

router = APIRouter(tags=["ingestion-jobs"])


def job_out(j: Job, filename: str | None = None) -> dict:
    return {"id": j.id, "project_id": j.project_id, "kind": j.kind, "subject_id": j.subject_id, "status": j.status,
            "attempts": j.attempts, "max_attempts": j.max_attempts, "progress": round(j.progress, 3),
            "stage": j.stage, "error": j.error, "filename": filename,
            "created_at": j.created_at.isoformat(),
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None}


def _filenames(db: Session, js: list[Job]) -> dict[str, str]:
    ids = [j.subject_id for j in js if j.kind == "ingest_document"]
    if not ids:
        return {}
    rows = db.execute(select(DocumentVersion.id, Document.filename).join(Document)
                      .where(DocumentVersion.id.in_(ids))).all()
    return dict(rows)


@router.get("/projects/{project_id}/ingestion-jobs")
def list_jobs(project_id: str, status: str | None = None, limit: int = Query(100, ge=1, le=1000),
              offset: int = Query(0, ge=0), user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    get_project(db, user, project_id)
    stmt = select(Job).where(Job.project_id == project_id)
    if status:
        stmt = stmt.where(Job.status == status)
    js = db.scalars(stmt.order_by(Job.created_at.desc()).offset(offset).limit(limit)).all()
    names = _filenames(db, js)
    return {"items": [job_out(j, names.get(j.subject_id)) for j in js]}


def _load(db: Session, user: User, job_id: str, role: str = "viewer") -> Job:
    j = db.get(Job, job_id)
    if j is None:
        raise HTTPException(404, "Job not found")
    get_project(db, user, j.project_id, role)
    return j


@router.get("/ingestion-jobs/{job_id}")
def read_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    j = _load(db, user, job_id)
    return job_out(j, _filenames(db, [j]).get(j.subject_id))


@router.post("/ingestion-jobs/{job_id}/cancel")
def cancel_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    return job_out(jobs.cancel(db, _load(db, user, job_id, "editor")))


@router.post("/ingestion-jobs/{job_id}/retry")
def retry_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    j = _load(db, user, job_id, "editor")
    try:
        return job_out(jobs.retry(db, j))
    except ValueError as e:
        raise HTTPException(409, str(e)) from None


@router.get("/projects/{project_id}/ingestion-jobs-stream")
async def stream_jobs(project_id: str, token: str) -> StreamingResponse:
    """Server-Sent Events: a snapshot of this project's active and recent jobs whenever it changes.
    EventSource cannot send headers, so the access token is passed as ?token=."""
    try:
        uid = jwt.decode(token, settings.secret_key, algorithms=["HS256"])["sub"]
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token") from None
    with dbmod.SessionLocal() as db:
        project = db.get(Project, project_id)
        if project is None or workspace_role(db, uid, project.workspace_id) is None:
            raise HTTPException(404, "Project not found")

    async def gen():
        last = None
        while True:
            with dbmod.SessionLocal() as db:
                js = db.scalars(select(Job).where(Job.project_id == project_id)
                                .order_by(Job.created_at.desc()).limit(200)).all()
                snap = [job_out(j, _filenames(db, js).get(j.subject_id)) for j in js]
            payload = json.dumps(snap)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
