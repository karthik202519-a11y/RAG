"""Celery entry point (Docker profile): `celery -A studio.jobs.celery_app worker`."""
from celery import Celery

from ..config import settings
from . import eval_job, figures_job, index_job, tasks  # noqa: F401  (registers job handlers)
from .runner import run_job

celery_app = Celery("studio", broker=settings.redis_url or "memory://", backend=None)
celery_app.conf.update(task_acks_late=True, worker_prefetch_multiplier=1, task_track_started=False)


@celery_app.task(name="studio.run_job")
def run_job_task(job_id: str) -> None:
    run_job(job_id)
