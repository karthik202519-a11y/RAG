"""Job handler: run an evaluation dataset against an index."""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from sqlalchemy import select

from .. import db as dbmod
from ..config import settings
from ..evaluation import service as ev
from ..models import EvalQuestion, EvalRun, Index
from .runner import CANCEL_HOOKS, DEAD_HOOKS, JobContext, PermanentJobError, handler


@handler("run_eval")
def run_eval(ctx: JobContext) -> None:
    db, job = ctx.db, ctx.job
    run = db.get(EvalRun, job.subject_id)
    if run is None:
        raise PermanentJobError("Evaluation run no longer exists")
    if run.status == "done":
        return
    index = db.get(Index, run.index_id)
    if index is None or index.status != "ready":
        raise PermanentJobError("The index is gone or not ready")
    cfg = run.config
    stmt = select(EvalQuestion).where(EvalQuestion.dataset_id == run.dataset_id, EvalQuestion.approved.is_(True))
    if cfg.get("split") and cfg["split"] != "all":
        stmt = stmt.where(EvalQuestion.split == cfg["split"])
    qs = db.scalars(stmt.order_by(EvalQuestion.position)).all()
    if not qs:
        raise PermanentJobError("No approved questions match this run's split")
    run.status, run.error = "running", None
    db.commit()
    t0, per_q = time.perf_counter(), []
    index_id = index.id
    snapshot = [(q.id, q.question, q.qtype, q.answerable, q.split) for q in qs]

    def one(qid: str):
        """Each question runs on its own database session (sessions are not shared between threads)."""
        with dbmod.SessionLocal() as sess:
            idx, q = sess.get(Index, index_id), sess.get(EvalQuestion, qid)
            try:
                return ev.run_one(sess, idx, q, cfg)
            except Exception as e:  # one failing question must not sink the run; it is recorded
                if isinstance(e, (MemoryError, KeyboardInterrupt)):
                    raise
                return {"question_id": q.id, "question": q.question, "qtype": q.qtype, "answerable": q.answerable, "split": q.split,
                        "metrics": ev.M.evaluate_question([], [], [], cfg["k"]) | {"hit": None, "first_rank": None},
                        "latency_ms": 0.0, "top1_dense_score": None, "llm_tokens": 0, "top": [],
                        "error": f"{type(e).__name__}: {str(e)[:200]}"}

    # Questions are independent, so several run at once; results keep the dataset's order.
    pool = ThreadPoolExecutor(max_workers=max(1, min(settings.eval_workers, len(snapshot))), thread_name_prefix="eval")
    try:
        futures = [pool.submit(one, s[0]) for s in snapshot]
        for n, f in enumerate(futures):
            while True:
                if ctx.cancelled():
                    raise InterruptedError("cancelled")
                try:
                    per_q.append(f.result(timeout=1.0))
                    break
                except TimeoutError:
                    continue
            ctx.progress((n + 1) / len(snapshot), f"question {n + 1}/{len(snapshot)}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    errors = [p for p in per_q if p.get("error")]
    if errors and len(errors) == len(per_q):
        raise PermanentJobError(f"Every question failed. First error: {errors[0]['error']}")
    run.per_question = per_q
    run.metrics = ev.summarize([p for p in per_q if not p.get("error")], cfg) | {"failed_questions": len(errors)}
    run.seconds, run.status = round(time.perf_counter() - t0, 2), "done"
    db.commit()


def _failed(db, job, error: str) -> None:
    run = db.get(EvalRun, job.subject_id)
    if run is not None:
        run.status, run.error = "failed", error
        db.commit()


DEAD_HOOKS["run_eval"] = _failed
CANCEL_HOOKS["run_eval"] = lambda db, job: _failed(db, job, "Run was cancelled.")
