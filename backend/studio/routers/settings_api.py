"""Read-only view of how this deployment is configured, plus live connection tests.

Credentials live in the environment (.env), never in the database, so this page shows which settings are present
as set / not set and never returns a value. Tests make one tiny real call and report what came back.
"""
import time
import uuid
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import __version__
from ..artifacts import get_store
from ..config import settings
from ..db import get_db
from ..embeddings import PROVIDERS
from ..generation import service as gen
from ..models import User
from ..security import current_user
from ..vectorstores import STORES, faiss_store, qdrant_store

router = APIRouter(prefix="/settings", tags=["settings"])


def _is_set(v: str) -> str:
    return "set" if v else "not set"


@router.get("/overview")
def overview(user: User = Depends(current_user)) -> dict:
    db_kind = "PostgreSQL" if settings.database_url.startswith("postgres") else "SQLite"
    return {
        "version": __version__,
        "runtime": [
            {"name": "Database", "value": db_kind, "env": "DATABASE_URL"},
            {"name": "Artifact storage", "value": "S3 / MinIO" if settings.s3_endpoint else "Local folder", "env": "S3_ENDPOINT"},
            {"name": "Job backend", "value": f"{settings.job_backend} ({settings.job_workers} workers)", "env": "STUDIO_JOB_BACKEND"},
            {"name": "Qdrant", "value": qdrant_store.mode(), "env": "QDRANT_URL"},
            {"name": "FAISS", "value": faiss_store.mode(), "env": None},
            {"name": "Max upload", "value": f"{settings.max_upload_mb} MB", "env": "STUDIO_MAX_UPLOAD_MB"},
            {"name": "Parallelism", "value": f"embed {settings.embed_concurrency} · eval {settings.eval_workers} · vision {settings.vision_workers}", "env": "STUDIO_*_WORKERS"},
        ],
        "azure": [
            {"env": "AZURE_OPENAI_ENDPOINT", "state": _is_set(settings.azure_endpoint)},
            {"env": "AZURE_OPENAI_API_KEY", "state": _is_set(settings.azure_api_key)},
            {"env": "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "state": _is_set(settings.azure_embedding_deployment), "value": settings.azure_embedding_deployment or None},
            {"env": "AZURE_OPENAI_CHAT_DEPLOYMENT", "state": _is_set(settings.azure_chat_deployment), "value": settings.azure_chat_deployment or None},
            {"env": "AZURE_OPENAI_CHAT_DEPLOYMENTS", "state": _is_set(settings.azure_chat_deployments), "value": settings.azure_chat_deployments or None},
            {"env": "AZURE_OPENAI_VISION_DEPLOYMENT", "state": _is_set(settings.azure_vision_deployment), "value": settings.azure_vision_deployment or None},
        ],
        "embedding": [{"id": p.id, "model": p.model, "available": p.status()[0], "detail": p.status()[1]} for p in PROVIDERS.values()],
        "generation": {"configured": gen.status_for(gen.default_model())[0], "detail": gen.status_for(gen.default_model())[1], "models": gen.configured_models()},
        "vision": {"configured": gen.vision_status()[0], "detail": gen.vision_status()[1]},
        "notes": ["Secrets are read from the environment and are never stored in the database or returned by the API.",
                  "Edit .env and restart the API to change a setting."],
    }


class TestIn(BaseModel):
    target: Literal["database", "storage", "embedding", "chat", "vision", "qdrant", "faiss", "chroma", "lancedb", "pgvector"]
    model: str | None = None


@router.post("/test")
def test_connection(body: TestIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    t0 = time.perf_counter()
    try:
        detail = _run(body, db)
        ok = True
    except Exception as e:  # a failed test is the answer, not a server error
        ok, detail = False, f"{type(e).__name__}: {str(e)[:240]}"
    return {"target": body.target, "ok": ok, "detail": detail, "ms": round((time.perf_counter() - t0) * 1000, 1)}


def _run(body: TestIn, db: Session) -> str:
    t = body.target
    if t == "database":
        db.execute(text("SELECT 1"))
        return "Query succeeded."
    if t == "storage":
        store = get_store()
        store.put("healthcheck/ping", b"ok")
        assert store.get("healthcheck/ping") == b"ok"
        store.delete_prefix("healthcheck/")
        return "Wrote, read back and removed a test object."
    if t == "embedding":
        p = PROVIDERS["azure_openai"]
        ok, why = p.status()
        if not ok:
            raise RuntimeError(why)
        r = p.embed(["connection test"])
        return f"Embedded one text with {p.model}: {r.vectors.shape[1]} dimensions, {r.tokens} tokens billed."
    if t == "chat":
        model = body.model or gen.default_model()
        r = gen.chat_complete(model, [{"role": "user", "content": "Reply with the single word: ok"}], 0.0, 200)
        return f"{r.model} replied in {r.completion_tokens} tokens ({r.prompt_tokens} prompt)."
    if t == "vision":
        import pymupdf
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 32, 32), False)
        pix.set_rect(pix.irect, (200, 30, 30))
        r = gen.chat_vision(settings.azure_vision_deployment, "Name the colour of this image in one word.", pix.tobytes("png"), 100)
        return f"{r.model} answered: {r.text.strip()[:60]}"
    if t == "qdrant":
        n = len(qdrant_store.client().get_collections().collections)
        return f"Qdrant ({qdrant_store.mode()}) is reachable; {n} collections."
    store = STORES[t]
    ok, why = store.status()
    if not ok:
        raise RuntimeError(why)
    name = "healthcheck_" + uuid.uuid4().hex[:12]
    vec = np.ones((1, 4), dtype=np.float32) / 2
    try:
        store.create_collection(name, 4)
        store.upsert(name, ["0" * 32], vec, [{}])
        hit = store.search(name, vec[0], 1)
        if not hit:
            raise RuntimeError("Test vector was not returned")
        return f"{t} wrote and searched a test vector (score {hit[0].score:.2f})."
    finally:
        store.delete_collection(name)
