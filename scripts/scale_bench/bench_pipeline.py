"""End to end at scale through the real API: upload, parse, index with Azure embeddings, then retrieval quality on needle questions.

The corpora are nested (the 1M documents are a subset of the 3M, which are a subset of the 10M), so the same questions are asked of a
growing haystack and any drop in hit rate is caused by size alone. Questions are written by gpt-4o-mini from random passages of the 1M
corpus; each carries a distinctive keyword from its passage, so a hit means the right document was returned and keyword_coverage says
whether the right passage text was.

usage: python scripts/scale_bench/bench_pipeline.py http://localhost:8001 1000000 3000000 10000000
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(Path(__file__).parent)]
from corpus import KIND, PDFS, select  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
SIZES = [int(a) for a in sys.argv[2:]] or [1_000_000]
OUT = ROOT / "docs" / "scale"
http = httpx.Client(base_url=BASE, timeout=600)
EMAIL, PW = "scale@example.com", "scale-bench-pass"


def call(method, path, **kw):
    r = http.request(method, path, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code} {r.text[:300]}")
    return r.json() if r.content else None


def login():
    r = http.post("/auth/register", json={"email": EMAIL, "password": PW, "display_name": "Scale"})
    if r.status_code not in (200, 201, 409):
        raise RuntimeError(r.text)
    tok = call("POST", "/auth/login", json={"email": EMAIL, "password": PW})["token"]
    http.headers["Authorization"] = f"Bearer {tok}"


def corpus_files(target: int) -> list[tuple[str, str]]:
    """[(unique upload name, path)] for the corpus of about `target` characters; PDFs first."""
    files = [(f"real-{i}.pdf", p) for i, p in enumerate(PDFS) if os.path.exists(p) and target >= 2_000_000]
    for n, (p, ext, _) in enumerate(select(target)):
        files.append((f"{n:05d}_{os.path.basename(p)}", p))
    return files


def ingest(pid: str, files: list[tuple[str, str]]) -> dict:
    t0 = time.perf_counter()
    for i in range(0, len(files), 25):
        batch = files[i:i + 25]
        handles = [("files", (name, open(path, "rb"))) for name, path in batch]
        try:
            call("POST", f"/projects/{pid}/documents", files=handles)
        finally:
            for _, (_, fh) in handles:
                fh.close()
    total = len(files)
    while True:
        st = call("GET", f"/projects/{pid}/ingestion-stats")
        done = st["by_status"].get("ready", 0) + st["by_status"].get("failed", 0)
        print(f"   ingest {done}/{total}", end="\r", flush=True)
        if done >= total:
            break
        time.sleep(3)
    secs = time.perf_counter() - t0
    st = call("GET", f"/projects/{pid}/ingestion-stats")
    return {"files": total, "seconds": round(secs, 1), "files_per_second": round(total / secs, 2), "failed": st["by_status"].get("failed", 0),
            "stats": {k: st[k] for k in st if k in ("pages", "tokens", "tables", "chars")}}


def build_index(pid: str, name: str, provider: str, store: str = "qdrant") -> dict:
    t0 = time.perf_counter()
    r = call("POST", f"/projects/{pid}/indexes", json={"name": name, "provider": provider, "store": store,
                                                       "chunking": {"method": "recursive", "params": {"size": 400, "overlap": 40}}})
    while True:
        i = call("GET", f"/indexes/{r['id']}")
        if i["status"] in ("ready", "failed"):
            break
        j = i.get("job") or {}
        print(f"   index {i['status']} {j.get('stage', '')} {int(100 * (j.get('progress') or 0))}%   ", end="\r", flush=True)
        time.sleep(3)
    if i["status"] != "ready":
        raise RuntimeError(f"index failed: {i.get('error')}")
    return {"id": i["id"], "seconds": round(time.perf_counter() - t0, 1), "chunks": i["chunk_count"], "tokens_embedded": i["tokens_embedded"],
            "tokens_billed": i["tokens_billed"], "cache_hits": i["cache_hits"], "build_seconds": i["build_seconds"], "dimension": i["dimension"]}


def make_questions(index_id: str, n: int, seed: int = 7) -> list[dict]:
    """Questions from random substantial chunks of the smallest corpus, written by gpt-4o-mini; keyword = a rare identifier in the chunk."""
    from openai import OpenAI
    from studio.config import settings
    client = OpenAI(base_url=settings.azure_endpoint.rstrip("/") + "/openai/v1/", api_key=settings.azure_api_key)
    chunks = []
    off = 0
    while len(chunks) < 6000:
        page = call("GET", f"/indexes/{index_id}/chunks", params={"limit": 200, "offset": off})
        items = page["items"] if isinstance(page, dict) else page
        if not items:
            break
        chunks += items
        off += 200
    pool = [c for c in chunks if 150 <= c["tokens"] <= 400 and c["type"] == "text" and not c["filename"].endswith(".json")]
    random.Random(seed).shuffle(pool)
    qs, seen = [], {}
    for c in pool:
        if len(qs) >= n:
            break
        if seen.get(c["filename"], 0) >= 8:
            continue
        r = client.chat.completions.create(model=settings.azure_chat_deployment, temperature=0, max_completion_tokens=200, messages=[
            {"role": "system", "content": "Write ONE specific question that can be answered only from the passage. Do not copy a whole sentence; do not mention 'the passage'. "
                                          'Reply as JSON: {"question": "...", "keyword": "a rare identifier, name or number from the passage that the answer depends on"}'},
            {"role": "user", "content": c["text"][:2500]}])
        try:
            j = json.loads(r.choices[0].message.content[r.choices[0].message.content.index("{"):r.choices[0].message.content.rindex("}") + 1])
        except Exception:
            continue
        kw = str(j.get("keyword", "")).strip()
        if kw and kw.lower() in c["text"].lower() and len(j.get("question", "")) > 15:
            seen[c["filename"]] = seen.get(c["filename"], 0) + 1
            qs.append({"question": j["question"], "qtype": KIND.get(c["filename"].rsplit(".", 1)[-1].lower(), "pdf"), "evidence": [{"document": c["filename"], "pages": c["pages"] or [1]}],
                       "keywords": [kw], "answerable": True, "split": "test"})
    return qs


def evaluate(pid: str, index_id: str, questions: list[dict], label: str) -> dict:
    ds = call("POST", f"/projects/{pid}/eval-datasets", json={"name": f"needles {label}", "description": "scale test"})
    call("POST", f"/eval-datasets/{ds['id']}/import", json={"questions": questions, "format": "native", "skip_missing_documents": True})
    out = {}
    for mode in ("dense", "hybrid"):
        run = call("POST", f"/eval-datasets/{ds['id']}/runs", json={"index_id": index_id, "mode": mode, "k": 5, "label": f"{label} {mode}"})
        while True:
            r = call("GET", f"/eval-runs/{run['id']}")
            if r["status"] in ("done", "failed"):
                break
            time.sleep(2)
        m = r["metrics"] or {}
        ret = m.get("retrieval", {})
        out[mode] = {"questions": m.get("answerable"), "hit": ret.get("hit", {}).get("mean"), "mrr": ret.get("mrr", {}).get("mean"), "ndcg": ret.get("ndcg", {}).get("mean"),
                     "recall": ret.get("recall", {}).get("mean"), "keyword_coverage": ret.get("keyword_coverage", {}).get("mean"),
                     "latency_p50_ms": m.get("latency_ms", {}).get("p50"), "latency_p95_ms": m.get("latency_ms", {}).get("p95"), "hit_ci95": ret.get("hit", {}).get("ci95")}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    login()
    ws = call("GET", "/workspaces")[0]
    results = {"base": BASE, "sizes": {}}
    questions = None
    for target in SIZES:
        print(f"\n=== {target:,} characters", flush=True)
        pid = call("POST", "/projects", json={"workspace_id": ws["id"], "name": f"scale {target}"})["id"]
        files = corpus_files(target)
        print(f"   {len(files)} files", flush=True)
        ing = ingest(pid, files)
        print(f"\n   ingested in {ing['seconds']}s ({ing['files_per_second']} files/s), failed {ing['failed']}", flush=True)
        idx = build_index(pid, f"recursive-400 azure {target}", "azure_openai")
        print(f"\n   index: {idx['chunks']:,} chunks, {idx['tokens_embedded']:,} tokens, billed {idx['tokens_billed']:,}, cache hits {idx['cache_hits']:,}, {idx['seconds']}s", flush=True)
        if questions is None:
            questions = make_questions(idx["id"], 40)
            (OUT / "needle_questions.json").write_text(json.dumps(questions, indent=1), encoding="utf-8")
            print(f"   wrote {len(questions)} needle questions", flush=True)
        ev = evaluate(pid, idx["id"], questions, f"{target}")
        print("   retrieval:", json.dumps({k: {a: v[a] for a in ("hit", "mrr", "keyword_coverage", "latency_p50_ms")} for k, v in ev.items()}), flush=True)
        results["sizes"][str(target)] = {"project_id": pid, "ingest": ing, "index": idx, "retrieval": ev}
        (OUT / "pipeline.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
