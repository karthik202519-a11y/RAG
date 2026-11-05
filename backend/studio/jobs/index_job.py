"""Job handler: build an index (chunk -> embed -> store vectors) from a frozen configuration."""
import json
import time

from ..artifacts import get_store
from ..chunking import ChunkingError, run_chunking
from ..chunking.context import use_embedder
from ..embeddings import NotConfigured, embed_texts, get_provider
from ..chunking.tokens import count as count_tokens
from ..models import Document, DocumentVersion, Index, IndexChunk, IndexParent, uid
from ..vectorstores import store_for
from .runner import CANCEL_HOOKS, DEAD_HOOKS, JobContext, PermanentJobError, handler

SLICE = 128  # chunks embedded and written per step; also the cancellation granularity


def embed_input(chunk_text: str, ctype: str, section: str | None, section_prefix: bool, context: str | None = None) -> str:
    """What the model sees. Text chunks get their section title as context; tables already carry it.
    A contextual index also prepends the model-written sentence stored on the chunk."""
    body = f"Section: {section}\n{chunk_text}" if section_prefix and ctype == "text" and section else chunk_text
    return f"{context}\n{body}" if context else body


def _add_context(ctx: JobContext, db, rows: list, model: str) -> None:
    """Write a situating sentence for every text chunk, in parallel. A chunk whose call fails simply goes without one."""
    from concurrent.futures import ThreadPoolExecutor

    from ..chunking.contextual import situate
    from ..config import settings
    targets = [n for n, r in enumerate(rows) if r.ctype == "text"]
    ctx.progress(0.22, f"writing context for {len(targets)} chunks")
    jobs_in = []
    for n in targets:
        r = rows[n]
        prev = rows[n - 1].text if n > 0 and rows[n - 1].document_id == r.document_id else ""
        nxt = rows[n + 1].text if n + 1 < len(rows) and rows[n + 1].document_id == r.document_id else ""
        jobs_in.append((n, r.filename, r.section, prev, r.text, nxt))

    def one(a):
        try:
            return a[0], situate(a[1], a[2], a[3], a[4], a[5], model)
        except Exception:
            return a[0], None

    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, min(settings.eval_workers, len(jobs_in) or 1)), thread_name_prefix="ctx") as pool:
        for done, (n, out) in enumerate(pool.map(one, jobs_in), 1):
            if ctx.cancelled():
                raise InterruptedError("cancelled")
            if out is None:
                failed += 1
            else:
                rows[n].context = out[0]
            if done % 20 == 0:
                ctx.progress(0.22 + 0.03 * done / len(jobs_in), f"context {done}/{len(jobs_in)}")
    if failed == len(jobs_in) and jobs_in:
        raise PermanentJobError("The chat model failed for every chunk, so no context could be written")
    db.commit()


@handler("build_index")
def build_index(ctx: JobContext) -> None:
    db, job = ctx.db, ctx.job
    idx = db.get(Index, job.subject_id)
    if idx is None:
        raise PermanentJobError("Index no longer exists")
    if idx.status == "ready" and vs.collection_dim(idx.collection) == idx.dimension:
        return  # idempotent
    vs = store_for(idx.store)
    cfg = idx.config
    provider = get_provider(idx.provider)
    ok, why = provider.status()
    if not ok:
        raise PermanentJobError(f"Embedding provider '{idx.provider}' is not configured: {why}")

    # A retry starts clean so a half-built index can never be searched.
    idx.status, idx.error = "building", None
    db.query(IndexChunk).filter(IndexChunk.index_id == idx.id).delete()
    db.commit()
    vs.delete_collection(idx.collection)

    t0 = time.perf_counter()
    ctx.progress(0.02, "chunking")
    store, rows, all_parents = get_store(), [], []
    versions = cfg["version_ids"]
    for n, vid in enumerate(versions):
        ver = db.get(DocumentVersion, vid)
        if ver is None or not ver.extraction_key:
            raise PermanentJobError(f"Document version {vid} is no longer available")
        doc = db.get(Document, ver.document_id)
        pages = json.loads(store.get(ver.extraction_key))
        try:
            with use_embedder(lambda prov, texts: embed_texts(db, prov, texts).vectors):
                res = run_chunking(pages, cfg["chunking"]["method"], cfg["chunking"]["params"])
        except ChunkingError as e:
            raise PermanentJobError(f"Chunking failed for {doc.filename}: {e}") from e
        except NotConfigured as e:
            raise PermanentJobError(str(e)) from e
        parents: dict[tuple, IndexParent] = {}
        for c in res["chunks"]:
            pid = None
            if c.get("parent_start") is not None:  # parent-child chunking: remember the larger passage
                key = (c["parent_start"], c["parent_end"])
                if key not in parents:
                    ptxt = res["doc_text"][c["parent_start"]:c["parent_end"]]
                    parents[key] = IndexParent(id=uid(), index_id=idx.id, document_id=doc.id, version_id=ver.id, filename=doc.filename,
                                               text=ptxt, tokens=count_tokens(ptxt), pages=[], section=c["section"])
                parents[key].pages = sorted(set(parents[key].pages) | set(c["pages"]))
                pid = parents[key].id
            rows.append(IndexChunk(index_id=idx.id, document_id=doc.id, version_id=ver.id, filename=doc.filename,
                                   chunk_index=c["index"], start=c["start"], end=c["end"], text=c["text"],
                                   ctype=c["type"], tokens=c["tokens"], pages=c["pages"],
                                   printed_pages=c["printed_pages"], section=c["section"], parent_id=pid))
        all_parents.extend(parents.values())
        ctx.progress(0.02 + 0.2 * (n + 1) / len(versions), f"chunking {n + 1}/{len(versions)} documents")
    if not rows:
        raise PermanentJobError("The chosen documents produced no chunks")
    db.add_all(all_parents + rows)
    db.commit()
    if cfg.get("contextual"):
        _add_context(ctx, db, rows, cfg["contextual"]["model"])

    billed = hits = embedded_tokens = 0
    dim = None
    for i in range(0, len(rows), SLICE):
        if ctx.cancelled():
            raise InterruptedError("cancelled")
        part = rows[i:i + SLICE]
        try:
            res = embed_texts(db, idx.provider, [embed_input(r.text, r.ctype, r.section, cfg["section_prefix"], r.context) for r in part])
        except NotConfigured as e:
            raise PermanentJobError(str(e)) from e
        if dim is None:
            dim = int(res.vectors.shape[1])
            vs.create_collection(idx.collection, dim, idx.metric)
        elif res.vectors.shape[1] != dim:
            raise PermanentJobError("Provider returned vectors of inconsistent size")
        vs.upsert(idx.collection, [r.id for r in part], res.vectors,
                            [{"document_id": r.document_id, "ctype": r.ctype, "chunk_index": r.chunk_index} for r in part])
        billed += res.tokens_billed
        hits += res.cache_hits
        embedded_tokens += sum(r.tokens for r in part)
        ctx.progress(0.25 + 0.72 * min(1.0, (i + SLICE) / len(rows)), f"embedding {min(i + SLICE, len(rows))}/{len(rows)} chunks")

    stored = vs.count(idx.collection)
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
