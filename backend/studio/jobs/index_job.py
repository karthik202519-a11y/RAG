"""Job handler: build an index (chunk -> embed -> store vectors) from a frozen configuration."""
import json
import time

from ..artifacts import get_store
from ..chunking import ChunkingError, run_chunking
from ..embeddings import NotConfigured, embed_texts, get_provider
from ..models import Document, DocumentVersion, Index, IndexChunk
from ..vectorstores import qdrant_store
from .runner import CANCEL_HOOKS, DEAD_HOOKS, JobContext, PermanentJobError, handler

SLICE = 128  # chunks embedded and written per step; also the cancellation granularity


def embed_input(chunk_text: str, ctype: str, section: str | None, section_prefix: bool) -> str:
    """What the model sees. Text chunks get their section title as context; tables already carry it."""
    return f"Section: {section}\n{chunk_text}" if section_prefix and ctype == "text" and section else chunk_text


@handler("build_index")
def build_index(ctx: JobContext) -> None:
    db, job = ctx.db, ctx.job
    idx = db.get(Index, job.subject_id)
    if idx is None:
        raise PermanentJobError("Index no longer exists")
    if idx.status == "ready" and qdrant_store.collection_dim(idx.collection) == idx.dimension:
        return  # idempotent
    cfg = idx.config
    provider = get_provider(idx.provider)
    ok, why = provider.status()
    if not ok:
        raise PermanentJobError(f"Embedding provider '{idx.provider}' is not configured: {why}")

    # A retry starts clean so a half-built index can never be searched.
    idx.status, idx.error = "building", None
    db.query(IndexChunk).filter(IndexChunk.index_id == idx.id).delete()
    db.commit()
    qdrant_store.delete_collection(idx.collection)

    t0 = time.perf_counter()
    ctx.progress(0.02, "chunking")
    store, rows = get_store(), []
    versions = cfg["version_ids"]
    for n, vid in enumerate(versions):
        ver = db.get(DocumentVersion, vid)
        if ver is None or not ver.extraction_key:
            raise PermanentJobError(f"Document version {vid} is no longer available")
        doc = db.get(Document, ver.document_id)
        pages = json.loads(store.get(ver.extraction_key))
        try:
            res = run_chunking(pages, cfg["chunking"]["method"], cfg["chunking"]["params"])
        except ChunkingError as e:
            raise PermanentJobError(f"Chunking failed for {doc.filename}: {e}") from e
        for c in res["chunks"]:
            rows.append(IndexChunk(index_id=idx.id, document_id=doc.id, version_id=ver.id, filename=doc.filename,
                                   chunk_index=c["index"], start=c["start"], end=c["end"], text=c["text"],
                                   ctype=c["type"], tokens=c["tokens"], pages=c["pages"],
                                   printed_pages=c["printed_pages"], section=c["section"]))
        ctx.progress(0.02 + 0.2 * (n + 1) / len(versions), f"chunking {n + 1}/{len(versions)} documents")
    if not rows:
        raise PermanentJobError("The chosen documents produced no chunks")
    db.add_all(rows)
    db.commit()

    billed = hits = embedded_tokens = 0
    dim = None
    for i in range(0, len(rows), SLICE):
        if ctx.cancelled():
            raise InterruptedError("cancelled")
        part = rows[i:i + SLICE]
        try:
            res = embed_texts(db, idx.provider, [embed_input(r.text, r.ctype, r.section, cfg["section_prefix"]) for r in part])
        except NotConfigured as e:
            raise PermanentJobError(str(e)) from e
        if dim is None:
            dim = int(res.vectors.shape[1])
            qdrant_store.create_collection(idx.collection, dim, idx.metric)
        elif res.vectors.shape[1] != dim:
            raise PermanentJobError("Provider returned vectors of inconsistent size")
        qdrant_store.upsert(idx.collection, [r.id for r in part], res.vectors,
                            [{"document_id": r.document_id, "ctype": r.ctype, "chunk_index": r.chunk_index} for r in part])
        billed += res.tokens_billed
        hits += res.cache_hits
        embedded_tokens += sum(r.tokens for r in part)
        ctx.progress(0.25 + 0.72 * min(1.0, (i + SLICE) / len(rows)), f"embedding {min(i + SLICE, len(rows))}/{len(rows)} chunks")

    stored = qdrant_store.count(idx.collection)
    if stored != len(rows):
        raise RuntimeError(f"Vector store holds {stored} vectors but {len(rows)} chunks were written")
    idx.dimension, idx.chunk_count, idx.tokens_embedded = dim, len(rows), embedded_tokens
    idx.tokens_billed, idx.cache_hits = billed, hits
    idx.build_seconds, idx.status = round(time.perf_counter() - t0, 2), "ready"
    db.commit()


def _failed(db, job, error: str) -> None:
    idx = db.get(Index, job.subject_id)
    if idx is not None:
        idx.status, idx.error = "failed", error
        db.commit()


def _cancelled(db, job) -> None:
    _failed(db, job, "Build was cancelled. Retry the job to build this index.")


DEAD_HOOKS["build_index"] = _failed
CANCEL_HOOKS["build_index"] = _cancelled
