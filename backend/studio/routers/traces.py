"""Traces: one row per recorded search, answer or background job, with its timed steps.

Spans are the stage timings the code measured while running; they are shown in the order recorded and are not
reconstructed. Nothing is sampled or estimated.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Index, Job, RetrievalRun, User
from ..security import current_user, get_project

router = APIRouter(tags=["traces"])
LABELS = {"embed_query_ms": "Embed query", "vector_search_ms": "Vector search", "bm25_ms": "BM25 search",
          "retrieval_ms": "Retrieval", "llm_ms": "Language model"}


def _spans(t: dict) -> list[dict]:
    return [{"name": LABELS.get(k, k.removesuffix("_ms").replace("_", " ").capitalize()), "ms": float(v)}
            for k, v in t.items() if isinstance(v, (int, float))]


@router.get("/projects/{project_id}/traces")
def list_traces(project_id: str, kind: str | None = Query(None, pattern="^(search|answer|job)$"), limit: int = Query(50, ge=1, le=200),
                user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    get_project(db, user, project_id)
    items: list[dict] = []
    if kind in (None, "search", "answer"):
        rows = db.execute(select(RetrievalRun, Index.name).join(Index, Index.id == RetrievalRun.index_id)
                          .where(Index.project_id == project_id).order_by(RetrievalRun.created_at.desc()).limit(limit)).all()
        for r, iname in rows:
            is_answer = r.mode.startswith("answer:")
            if kind == "search" and is_answer or kind == "answer" and not is_answer:
                continue
            spans = _spans(r.timings_ms or {})
            items.append({"id": r.id, "kind": "answer" if is_answer else "search", "name": r.query[:140], "detail": f"{r.mode} on {iname}",
                          "at": r.created_at.isoformat(), "total_ms": round(sum(s["ms"] for s in spans), 1), "spans": spans,
                          "attrs": {k: v for k, v in (r.params or {}).items() if k in ("model", "abstained", "top_k", "prompt_tokens", "completion_tokens", "cost_usd") and v is not None},
                          "results": len(r.result_ids or [])})
    if kind in (None, "job"):
        for j in db.scalars(select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc()).limit(limit)):
            wait = (j.started_at - j.created_at).total_seconds() * 1000 if j.started_at else None
            run = (j.finished_at - j.started_at).total_seconds() * 1000 if j.started_at and j.finished_at else None
            spans = [{"name": "Queued", "ms": round(wait, 1)}] if wait is not None else []
            if run is not None:
                spans.append({"name": "Running", "ms": round(run, 1)})
            items.append({"id": j.id, "kind": "job", "name": j.kind, "detail": f"{j.status}" + (f" · attempt {j.attempts}/{j.max_attempts}" if j.attempts > 1 else ""),
                          "at": j.created_at.isoformat(), "total_ms": round(sum(s["ms"] for s in spans), 1), "spans": spans,
                          "attrs": {"status": j.status, **({"error": j.error[:200]} if j.error else {})}, "results": None})
    items.sort(key=lambda x: x["at"], reverse=True)
    items = items[:limit]
    done = [i["total_ms"] for i in items if i["kind"] != "job" and i["total_ms"]]
    done.sort()
    summary = {"count": len(items), "searches": sum(i["kind"] == "search" for i in items), "answers": sum(i["kind"] == "answer" for i in items),
               "jobs": sum(i["kind"] == "job" for i in items), "failed_jobs": sum(1 for i in items if i["kind"] == "job" and i["attrs"].get("status") in ("failed", "dead")),
               "p50_ms": done[len(done) // 2] if done else None, "p95_ms": done[min(len(done) - 1, int(len(done) * 0.95))] if done else None,
               "tokens": sum((i["attrs"].get("prompt_tokens") or 0) + (i["attrs"].get("completion_tokens") or 0) for i in items)}
    return {"items": items, "summary": summary}
