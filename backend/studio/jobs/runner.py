"""Durable background jobs.

State lives in the `jobs` table, not in memory: a job survives a restart (it is re-queued),
handlers are idempotent (a finished job is never re-run), failures retry with exponential
backoff up to `max_attempts`, then the job is marked `dead`.

Two executors share this logic:
  thread  - in-process thread pool (local default, no services needed)
  celery  - Celery workers on Redis (Docker profile; `celery -A studio.jobs.celery_app worker`)
"""
import logging
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import db as dbmod
from ..config import settings
from ..models import Job

log = logging.getLogger("studio.jobs")
BACKOFF_BASE_SECONDS = 2.0

Handler = Callable[["JobContext"], None]
HANDLERS: dict[str, Handler] = {}


def handler(kind: str):
    def deco(fn: Handler) -> Handler:
        HANDLERS[kind] = fn
        return fn
    return deco


class JobContext:
    """Passed to handlers: the job row, a session, and throttled progress / cancel helpers."""

    def __init__(self, db: Session, job: Job) -> None:
        self.db, self.job, self._last = db, job, 0.0

    def progress(self, fraction: float, stage: str | None = None) -> None:
        now = time.monotonic()
        if now - self._last < 0.5 and fraction < 1.0:
            return
        self._last = now
        self.job.progress = max(0.0, min(1.0, fraction))
        if stage:
            self.job.stage = stage
        self.db.commit()

    def cancelled(self) -> bool:
        self.db.refresh(self.job)
        return bool(self.job.cancel_requested)


_pool: ThreadPoolExecutor | None = None
_timers: set[threading.Timer] = set()


def _executor() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=settings.job_workers, thread_name_prefix="studio-job")
    return _pool


def _dispatch(job_id: str, delay: float = 0.0) -> None:
    if settings.job_backend == "celery":
        from .celery_app import run_job_task
        run_job_task.apply_async(args=[job_id], countdown=delay)
    elif delay > 0:
        t = threading.Timer(delay, lambda: _executor().submit(run_job, job_id))
        t.daemon = True
        _timers.add(t)
        t.start()
    else:
        _executor().submit(run_job, job_id)


def enqueue(db: Session, project_id: str, kind: str, subject_id: str, params: dict | None = None) -> Job:
    if kind not in HANDLERS:
        raise ValueError(f"No handler for job kind '{kind}'")
    job = Job(project_id=project_id, kind=kind, subject_id=subject_id, params=params or {},
              max_attempts=settings.job_max_attempts)
    db.add(job)
    db.commit()
    _dispatch(job.id)
    return job


def run_job(job_id: str) -> None:
    """Execute one attempt of a job. Safe to call twice: finished jobs are skipped."""
    with dbmod.SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None or job.status in ("succeeded", "cancelled", "dead", "running"):
            return
        if job.cancel_requested:
            job.status, job.finished_at = "cancelled", datetime.now(timezone.utc)
            db.commit()
            return
        job.status, job.attempts = "running", job.attempts + 1
        job.started_at, job.error = datetime.now(timezone.utc), None
        db.commit()
        try:
            HANDLERS[job.kind](JobContext(db, job))
        except InterruptedError:
            db.rollback()
            job = db.get(Job, job_id)
            hook = CANCEL_HOOKS.get(job.kind)
            if hook:  # update the subject first so nobody sees "cancelled" next to a stale "processing"
                hook(db, job)
            job.status, job.stage, job.finished_at = "cancelled", "cancelled", datetime.now(timezone.utc)
            db.commit()
            return
        except Exception as e:  # noqa: BLE001 - any handler failure is a job failure
            db.rollback()
            log.warning("job %s attempt %s failed: %s", job_id, job.attempts, e)
            job = db.get(Job, job_id)
            job.error = f"{type(e).__name__}: {e}"[:2000]
            if job.attempts >= job.max_attempts or isinstance(e, PermanentJobError):
                _on_dead(db, job, job.error, traceback.format_exc())
                job.status, job.finished_at = "dead", datetime.now(timezone.utc)
                db.commit()
            else:
                job.status = "queued"
                db.commit()
                _dispatch(job_id, delay=BACKOFF_BASE_SECONDS * 2 ** (job.attempts - 1))
            return
        job = db.get(Job, job_id)
        job.status, job.progress, job.stage = "succeeded", 1.0, "done"
        job.finished_at = datetime.now(timezone.utc)
        db.commit()


class PermanentJobError(Exception):
    """Raise from a handler when retrying cannot help (e.g. a corrupt or unsupported file)."""


def _on_dead(db: Session, job: Job, error: str, tb: str) -> None:
    """Update the subject before exposing terminal job status to readers."""
    log.error("job %s is dead: %s\n%s", job.id, error, tb)
    hook = DEAD_HOOKS.get(job.kind)
    if hook:
        hook(db, job, error)


DEAD_HOOKS: dict[str, Callable[[Session, Job, str], None]] = {}
CANCEL_HOOKS: dict[str, Callable[[Session, Job], None]] = {}


def cancel(db: Session, job: Job) -> Job:
    if job.status in ("succeeded", "cancelled", "dead"):
        return job
    job.cancel_requested = True
    if job.status == "queued":
        job.status, job.finished_at = "cancelled", datetime.now(timezone.utc)
    db.commit()
    return job


def retry(db: Session, job: Job) -> Job:
    """Re-queue a dead or cancelled job with a fresh attempt budget."""
    if job.status not in ("dead", "cancelled", "failed"):
        raise ValueError(f"Job is {job.status}; only dead or cancelled jobs can be retried")
    job.status, job.attempts, job.error, job.cancel_requested = "queued", 0, None, False
    job.progress, job.stage, job.finished_at = 0.0, None, None
    db.commit()
    _dispatch(job.id)
    return job


def recover_interrupted() -> int:
    """Called at startup: jobs left queued/running by a crash or restart are re-queued."""
    with dbmod.SessionLocal() as db:
        stuck = db.scalars(select(Job).where(Job.status.in_(("queued", "running")))).all()
        for j in stuck:
            j.status = "queued"
        db.commit()
        for j in stuck:
            _dispatch(j.id)
        return len(stuck)


def shutdown() -> None:
    for t in list(_timers):
        t.cancel()
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
