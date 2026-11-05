"""Job handlers. `ingest_document`: parse an uploaded file and store the extraction artifact."""
import json
import shutil
import tempfile
import time
from pathlib import Path

from ..artifacts import get_store
from ..chunking.canon import build_canon
from ..chunking.tokens import count
from ..models import Document, DocumentVersion
from ..parsing import parse_file, parser_for
from .runner import CANCEL_HOOKS, DEAD_HOOKS, JobContext, PermanentJobError, handler


@handler("ingest_document")
def ingest_document(ctx: JobContext) -> None:
    db, job = ctx.db, ctx.job
    ver = db.get(DocumentVersion, job.subject_id)
    if ver is None:
        raise PermanentJobError("Document version no longer exists")
    if ver.status == "ready" and ver.extraction_key and get_store().exists(ver.extraction_key):
        return  # idempotent: already extracted
    if parser_for(ver.ext) is None:
        raise PermanentJobError(f"Unsupported file type .{ver.ext}")

    ver.status, ver.error = "processing", None
    db.commit()
    ctx.progress(0.02, "reading")
    store = get_store()
    work = Path(tempfile.mkdtemp(prefix="studio-ingest-"))
    try:
        src = work / f"source.{ver.ext}"
        src.write_bytes(store.get(ver.source_key))
        doc_name = db.get(Document, ver.document_id).filename

        def progress(done: int, total: int) -> None:
            ctx.progress(0.05 + 0.85 * done / max(1, total), f"parsing page {done}/{total}")

        t0 = time.perf_counter()
        try:
            parser, pages = parse_file(src, ver.ext, doc_name, work / "visual_pages", progress, ctx.cancelled,
                                      ctx.job.params.get("parser"))
        except InterruptedError:
            raise
        except Exception as e:
            raise PermanentJobError(f"Could not parse the file: {type(e).__name__}: {e}") from e
        elapsed = time.perf_counter() - t0

        ctx.progress(0.92, "saving")
        prefix = f"projects/{ver.project_id}/versions/{ver.id}"
        for png in (work / "visual_pages").glob("*.png"):
            store.put(f"{prefix}/visual_pages/{png.name}", png.read_bytes())
        key = f"{prefix}/extraction.json"
        store.put(key, json.dumps(pages, ensure_ascii=False).encode("utf-8"))

        canon = build_canon(pages)
        ver.parser, ver.extraction_key, ver.extraction_seconds = parser, key, round(elapsed, 2)
        ver.pages = len(pages)
        ver.char_count = len(canon.text)
        ver.token_count = count(canon.text)
        ver.table_count = sum(1 for p in pages for i in p["items"] if i["type"] == "table")
        ver.status = "ready"
        db.commit()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _mark_failed(db, job, error: str) -> None:
    ver = db.get(DocumentVersion, job.subject_id)
    if ver is not None:
        ver.status, ver.error = "failed", error
        db.commit()


DEAD_HOOKS["ingest_document"] = _mark_failed


def _mark_cancelled(db, job) -> None:
    """A cancelled parse leaves the file stored but not extracted: back to 'uploaded', retryable."""
    ver = db.get(DocumentVersion, job.subject_id)
    if ver is not None and ver.status != "ready":
        ver.status, ver.error = "uploaded", "Parsing was cancelled. Retry the job to parse this file."
        db.commit()


CANCEL_HOOKS["ingest_document"] = _mark_cancelled
