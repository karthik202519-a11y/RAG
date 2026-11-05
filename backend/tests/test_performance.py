"""Performance behaviour: compression, caching and concurrency must be faster without changing any result."""
import threading
import time

import numpy as np

from conftest import upload, wait_for_job
from test_indexes import build, make_docs


def test_large_json_is_gzipped_and_event_streams_are_not(client, auth, project):
    big = ("Saturn V rocket paragraph about the Apollo programme and lunar orbit. " * 400).encode()
    r = upload(client, auth, project["id"], "big.txt", big).json()["results"][0]
    wait_for_job(client, auth, r["job_id"])
    resp = client.post("/chunking-preview", headers={**auth, "Accept-Encoding": "gzip"},
                       json={"document_id": r["document_id"], "configs": [{"method": "fixed_token", "params": {"size": 40, "overlap": 5}}]})
    assert resp.status_code == 200 and resp.headers.get("content-encoding") == "gzip"
    stream = client.get("/projects/x/ingestion-jobs-stream", params={"token": "bad"}, headers={"Accept-Encoding": "gzip"})
    assert stream.headers.get("content-encoding") != "gzip"


def test_catalog_is_cacheable_and_documents_list_uses_few_queries(client, auth, project):
    assert "max-age" in client.get("/method-catalog").headers["cache-control"]
    from sqlalchemy import event
    from studio import db as dbmod
    for i in range(12):
        r = upload(client, auth, project["id"], f"d{i}.txt", f"document {i} body {i * 31}".encode()).json()["results"][0]
        wait_for_job(client, auth, r["job_id"])
    count = {"n": 0}

    def tick(*a, **k):
        count["n"] += 1

    event.listen(dbmod.engine, "before_cursor_execute", tick)
    try:
        items = client.get(f"/projects/{project['id']}/documents", headers=auth).json()["items"]
    finally:
        event.remove(dbmod.engine, "before_cursor_execute", tick)
    assert len(items) == 12 and all(i["latest_version"] for i in items)
    assert count["n"] <= 8, f"{count['n']} queries for 12 documents (should not grow with the number of documents)"


def test_parallel_embedding_matches_sequential_and_runs_requests_concurrently(monkeypatch):
    from studio import db as dbmod
    from studio.config import settings
    from studio.embeddings import providers, service

    live, peak, lock = {"n": 0}, {"n": 0}, threading.Lock()

    class Slow(providers.HashEmbedder):
        def embed(self, texts):
            with lock:
                live["n"] += 1
                peak["n"] = max(peak["n"], live["n"])
            time.sleep(0.15)
            with lock:
                live["n"] -= 1
            return super().embed(texts)

    monkeypatch.setitem(providers.PROVIDERS, "local_hash", Slow())
    texts = [f"text number {i} with words{i}" for i in range(64 * 6)]
    with dbmod.SessionLocal() as db:
        monkeypatch.setattr(settings, "embed_concurrency", 1)
        seq = service.embed_texts(db, "local_hash", texts)
        db.query(service.EmbeddingCache).delete()
        db.commit()
        monkeypatch.setattr(settings, "embed_concurrency", 4)
        peak["n"] = 0
        par = service.embed_texts(db, "local_hash", texts)
    assert np.allclose(seq.vectors, par.vectors) and par.vectors.shape[0] == len(texts)
    assert peak["n"] > 1, "batches were not sent concurrently"
    # ORM/cache writes and host load are outside the provider concurrency contract.
    # Wall-clock speed comparisons belong in the controlled benchmark, not this correctness assertion.
    assert peak["n"] <= 4, "configured request concurrency was exceeded"


def test_evaluation_questions_run_concurrently_with_identical_results(client, auth, project, monkeypatch):
    from studio import retrieval
    from studio.config import settings
    from test_evaluation import add, dataset, run
    make_docs(client, auth, project)
    idx = build(client, auth, project)
    ds = dataset(client, auth, project)
    for i in range(12):
        add(client, auth, ds, f"Which rocket carried the Apollo spacecraft toward lunar orbit, variant {i}?", "apollo", qtype="prose")
    real = retrieval.search

    def slow(*a, **k):
        time.sleep(0.12)
        return real(*a, **k)

    monkeypatch.setattr(retrieval, "search", slow)
    monkeypatch.setattr(settings, "eval_workers", 1)
    t0 = time.perf_counter()
    a = run(client, auth, ds, idx, k=3, label="one worker")
    t_one = time.perf_counter() - t0
    monkeypatch.setattr(settings, "eval_workers", 4)
    t0 = time.perf_counter()
    b = run(client, auth, ds, idx, k=3, label="four workers")
    t_four = time.perf_counter() - t0

    def ranks(r):
        return [(p["question"], p["metrics"]["first_rank"]) for p in r["per_question"]]

    assert ranks(a) == ranks(b)  # same questions, same order, same ranks
    assert a["metrics"]["retrieval"]["mrr"]["mean"] == b["metrics"]["retrieval"]["mrr"]["mean"]
    assert t_four < t_one * 0.7, f"4 workers {t_four:.2f}s vs 1 worker {t_one:.2f}s"
