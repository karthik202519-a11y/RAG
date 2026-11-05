"""RAG Learning Studio API.   uvicorn studio.main:app --reload   ->   http://127.0.0.1:8000/docs"""
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from . import __version__, jobs
from . import db as dbmod
from .config import settings
from .registry import registry
from .routers import auth, catalog, chunking_api, documents, evaluation, explorer, graph_api, indexes, jobs_api, parser_lab, projects, settings_api, traces

log = logging.getLogger("studio")


def migrate() -> None:
    """Bring the database to the latest schema (Alembic). Runs at startup; set STUDIO_SKIP_MIGRATE=1 to skip."""
    cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    cfg.set_main_option("script_location", str(Path(__file__).resolve().parents[1] / "alembic"))
    url = dbmod.engine.url.render_as_string(hide_password=False)
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(cfg, "head")


def warm_up() -> None:
    """Pay the one-off start-up costs (tokenizer tables, vector store, PDF engine) before the first user request."""
    try:
        from .chunking.tokens import get_encoding
        from .vectorstores import qdrant_store
        get_encoding("cl100k_base")
        qdrant_store.client()
        import pymupdf  # noqa: F401
    except Exception:  # warming is best effort; a failure here must never stop the app
        log.exception("warm-up failed (ignored)")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if not os.getenv("STUDIO_SKIP_MIGRATE"):
        migrate()
    if not os.getenv("STUDIO_SKIP_WARMUP"):
        threading.Thread(target=warm_up, name="warm-up", daemon=True).start()  # do not delay start-up
    n = jobs.recover_interrupted()
    if n:
        log.info("re-queued %s interrupted job(s)", n)
    yield
    jobs.shutdown()


app = FastAPI(title="RAG Learning Studio", version=__version__, lifespan=lifespan,
              description="A RAG engineering laboratory: ingest, parse, chunk, embed, retrieve, evaluate.")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                   allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
# Large JSON (a chunk preview is ~700 KB) compresses about 6x. Event streams are excluded by Starlette's defaults.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
for r in (auth.router, projects.router, documents.router, jobs_api.router, catalog.router, chunking_api.router,
          indexes.router, explorer.router, evaluation.router, parser_lab.router, traces.router, settings_api.router, graph_api.router):
    app.include_router(r)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "version": __version__, "job_backend": settings.job_backend,
            "database": dbmod.engine.dialect.name, "artifact_store": "s3" if settings.s3_endpoint else "local",
            "methods": registry.categories()}
