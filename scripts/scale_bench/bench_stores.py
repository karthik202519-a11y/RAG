"""Load the same vectors into every vector store and compare build time, search latency and recall against exact search.

Uses the chunks of the largest index from bench_pipeline.py (their Azure embeddings come from the embedding cache, so this costs nothing).
Stop the port-8001 API first: embedded Qdrant takes a lock on its folder.

usage: STUDIO_DATA_DIR=data/scale/studio python scripts/scale_bench/bench_stores.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("STUDIO_DATA_DIR", str(ROOT / "data" / "scale" / "studio"))
sys.path[:0] = [str(ROOT / "backend")]

from sqlalchemy import select  # noqa: E402

from studio import db as dbmod  # noqa: E402
from studio.config import settings  # noqa: E402
from studio.embeddings import embed_texts  # noqa: E402
from studio.jobs.index_job import embed_input  # noqa: E402
from studio.models import Index, IndexChunk  # noqa: E402
from studio.vectorstores import STORES  # noqa: E402

PG = os.getenv("TEST_PGVECTOR_URL", "postgresql://postgres:studio@localhost:5434/vec")


def main() -> None:
    dbmod.init_engine() if hasattr(dbmod, "init_engine") else None
    with dbmod.SessionLocal() as db:
        idx = db.scalars(select(Index).where(Index.status == "ready").order_by(Index.chunk_count.desc())).first()
        rows = db.scalars(select(IndexChunk).where(IndexChunk.index_id == idx.id)).all()
        texts = [embed_input(r.text, r.ctype, r.section, True, r.context) for r in rows]
        t = time.perf_counter()
        res = embed_texts(db, idx.provider, texts)
        print(f"{len(rows):,} vectors from the cache in {time.perf_counter() - t:.1f}s (cache hits {res.cache_hits:,}, billed {res.tokens_billed:,})", flush=True)
    vecs = res.vectors.astype(np.float32)
    ids = [r.id for r in rows]
    meta = [{"document_id": r.document_id, "ctype": r.ctype, "chunk_index": r.chunk_index} for r in rows]
    rng = np.random.default_rng(3)
    pick = rng.choice(len(vecs), size=min(200, len(vecs)), replace=False)
    queries = vecs[pick] + rng.normal(0, 0.02, size=(len(pick), vecs.shape[1])).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    docs = sorted({m["document_id"] for m in meta})
    filt = random.Random(5).sample(docs, min(20, len(docs)))
    settings.pgvector_url = PG
    out = {"vectors": len(vecs), "dimension": int(vecs.shape[1]), "queries": len(queries), "stores": {}}
    truth: list[set] = []
    for name in ("faiss", "qdrant", "chroma", "lancedb", "pgvector"):
        store = STORES[name]
        coll = f"bench_{name}"
        ok, why = store.status()
        if not ok:
            out["stores"][name] = {"error": why}
            continue
        try:
            store.delete_collection(coll)
            t = time.perf_counter()
            store.create_collection(coll, vecs.shape[1], "cosine")
            for i in range(0, len(vecs), 500):
                store.upsert(coll, ids[i:i + 500], vecs[i:i + 500], meta[i:i + 500])
            load = time.perf_counter() - t
            assert store.count(coll) == len(vecs)
            lat, hits = [], []
            for q in queries:
                t = time.perf_counter()
                h = store.search(coll, q, 10)
                lat.append((time.perf_counter() - t) * 1000)
                hits.append({x.id for x in h})
            flat = []
            for q in queries[:60]:
                t = time.perf_counter()
                store.search(coll, q, 10, document_ids=filt)
                flat.append((time.perf_counter() - t) * 1000)
            if name == "faiss":
                truth = hits
            out["stores"][name] = {"load_seconds": round(load, 2), "p50_ms": round(float(np.percentile(lat, 50)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
                                   "recall_at_10": round(float(np.mean([len(a & b) / 10 for a, b in zip(hits, truth)])), 4) if truth else None,
                                   "filtered_p50_ms": round(float(np.percentile(flat, 50)), 2)}
            print(name, out["stores"][name], flush=True)
        except Exception as e:
            out["stores"][name] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            print(name, out["stores"][name], flush=True)
        finally:
            try:
                store.delete_collection(coll)
            except Exception:
                pass
    (ROOT / "docs" / "scale").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "scale" / "stores.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
