"""Job handler: extract a knowledge graph from an index's chunks with the chat model."""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from sqlalchemy import delete, select

from ..config import settings
from ..graph import extract_chunk
from ..models import GraphBuild, GraphEdge, GraphEntity, Index, IndexChunk
from .runner import CANCEL_HOOKS, DEAD_HOOKS, JobContext, PermanentJobError, handler

MIN_TOKENS = 25  # headings and fragments rarely hold a relation


def _pick(rows: list, n: int) -> list:
    """At most n chunks spread evenly across the whole index, so a small budget still covers every document."""
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


@handler("build_graph")
def build_graph(ctx: JobContext) -> None:
    db, job = ctx.db, ctx.job
    build = db.get(GraphBuild, job.subject_id)
    index = db.get(Index, job.subject_id)
    if build is None or index is None or index.status != "ready":
        raise PermanentJobError("The index is gone or not ready")
    max_chunks = int(job.params.get("max_chunks", 150))
    model = build.model
    rows = db.scalars(select(IndexChunk).where(IndexChunk.index_id == index.id, IndexChunk.tokens >= int(job.params.get("min_tokens", MIN_TOKENS)))
                      .order_by(IndexChunk.document_id, IndexChunk.chunk_index)).all()
    rows = _pick(list(rows), max_chunks)
    if not rows:
        raise PermanentJobError("This index has no chunks long enough to extract from")
    db.execute(delete(GraphEdge).where(GraphEdge.index_id == index.id))
    db.execute(delete(GraphEntity).where(GraphEntity.index_id == index.id))
    build.status, build.error = "running", None
    build.chunks_requested, build.chunks_processed, build.chunks_failed, build.relations_dropped, build.tokens = len(rows), 0, 0, 0, 0
    db.commit()
    snapshot = [(r.id, r.text) for r in rows]
    t0 = time.perf_counter()

    def one(item):
        try:
            return item[0], extract_chunk(item[1], model), None
        except Exception as e:  # one bad chunk must not sink the build
            return item[0], None, f"{type(e).__name__}: {str(e)[:160]}"

    entities: dict[str, GraphEntity] = {}
    seen_in: dict[str, set] = {}
    edges: list[tuple[str, str, str, str]] = []
    first_error = None
    pool = ThreadPoolExecutor(max_workers=max(1, min(settings.eval_workers, len(snapshot))), thread_name_prefix="graph")
    try:
        futures = [pool.submit(one, s) for s in snapshot]
        for n, f in enumerate(futures):
            while True:
                if ctx.cancelled():
                    raise InterruptedError("cancelled")
                try:
                    cid, out, err = f.result(timeout=1.0)
                    break
                except TimeoutError:
                    continue
            if out is None:
                build.chunks_failed += 1
                first_error = first_error or err
            else:
                build.chunks_processed += 1
                build.tokens += out["tokens"]
                build.relations_dropped += out["dropped"]
                for e in out["entities"]:
                    ent = entities.get(e["norm"])
                    if ent is None:
                        ent = entities[e["norm"]] = GraphEntity(index_id=index.id, name=e["name"], norm=e["norm"], etype=e["type"], mentions=0)
                    seen_in.setdefault(e["norm"], set()).add(cid)
                    ent.mentions = len(seen_in[e["norm"]])
                for r in out["relations"]:
                    edges.append((r["source"], r["target"], r["relation"], cid))
            ctx.progress((n + 1) / len(snapshot), f"chunk {n + 1}/{len(snapshot)}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if build.chunks_processed == 0:
        raise PermanentJobError(f"Every chunk failed. First error: {first_error}")
    db.add_all(entities.values())
    db.flush()
    seen_edges = set()
    for a, b, rel, cid in edges:
        key = (entities[a].id, entities[b].id, rel.lower(), cid)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        db.add(GraphEdge(index_id=index.id, src_id=key[0], dst_id=key[1], relation=rel, chunk_id=cid))
    build.seconds, build.status = round(time.perf_counter() - t0, 2), "ready"
    db.commit()


def _failed(db, job, error: str) -> None:
    b = db.get(GraphBuild, job.subject_id)
    if b is not None:
        b.status, b.error = "failed", error
        db.commit()


DEAD_HOOKS["build_graph"] = _failed
CANCEL_HOOKS["build_graph"] = lambda db, job: _failed(db, job, "Build was cancelled.")
