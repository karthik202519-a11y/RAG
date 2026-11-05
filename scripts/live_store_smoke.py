"""Live Azure + real vector-store API smoke test on isolated local application data.

Uses a tiny synthetic document, bills a few real embedding/chat tokens, and writes evidence.
No credentials, JWTs or generated account passwords are printed or persisted in the report.
"""
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(tempfile.mkdtemp(prefix="rag-live-stores-"))
os.environ["STUDIO_DATA_DIR"] = str(DATA)
os.environ["DATABASE_URL"] = f"sqlite:///{(DATA / 'app.db').as_posix()}"
os.environ["STUDIO_JOB_BACKEND"] = "thread"
os.environ["STUDIO_SKIP_WARMUP"] = "1"
os.environ["STUDIO_SECRET_KEY"] = uuid.uuid4().hex
os.environ["QDRANT_URL"] = ""
os.environ["QDRANT_PATH"] = str(DATA / "qdrant")
os.environ["S3_ENDPOINT"] = ""
os.environ["PGVECTOR_URL"] = os.getenv("TEST_PGVECTOR_URL", "postgresql://postgres:studio@localhost:5434/vec")
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient
from studio.main import app
from studio import jobs
from studio.config import settings
from studio.vectorstores import STORES


def wait(client, headers, job):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        result = client.get(f"/ingestion-jobs/{job}", headers=headers).json()
        if result["status"] in ("succeeded", "dead", "cancelled"):
            assert result["status"] == "succeeded", "background job failed"
            return
        time.sleep(0.1)
    raise RuntimeError("background job timed out")


def run():
    report = {"kind": "live Azure embeddings/chat and real vector stores; isolated synthetic document", "stores": []}
    path = ROOT / "output" / "live-store-smoke.json"
    path.parent.mkdir(exist_ok=True)
    if not (settings.azure_endpoint and settings.azure_api_key and settings.azure_embedding_deployment and settings.azure_chat_deployment):
        report["status"] = "blocked: Azure embedding/chat configuration is incomplete"
        path.write_text(json.dumps(report, indent=2))
        print(report["status"])
        return 2
    try:
        with TestClient(app) as client:
            r = client.post("/auth/register", json={"email": f"live-{uuid.uuid4().hex[:10]}@example.com", "password": uuid.uuid4().hex, "display_name": "Live verification"})
            assert r.status_code == 201
            headers = {"Authorization": "Bearer " + r.json()["token"]}
            workspace = client.get("/workspaces", headers=headers).json()[0]["id"]
            project = client.post("/projects", headers=headers, json={"workspace_id": workspace, "name": "Live store verification"}).json()["id"]
            body = b"Transaction ZX-4471 requires approval before deployment. The designated reviewer is Alice."
            uploaded = client.post(f"/projects/{project}/documents", headers=headers, files=[("files", ("approval.txt", body, "text/plain"))])
            assert uploaded.status_code == 201
            wait(client, headers, uploaded.json()["results"][0]["job_id"])
            for name, store in STORES.items():
                row = {"store": name, "status": "failed"}
                report["stores"].append(row)
                ok, _ = store.status()
                if not ok:
                    row["status"] = "blocked: store unavailable"
                    continue
                idx = None
                try:
                    r = client.post(f"/projects/{project}/indexes", headers=headers, json={"name": name, "provider": "azure_openai", "store": name, "chunking": {"method": "recursive", "params": {"size": 100, "overlap": 10}}})
                    assert r.status_code == 201, "index creation failed"
                    idx = r.json()
                    wait(client, headers, idx["job"]["id"])
                    base = f"/indexes/{idx['id']}"
                    details = client.get(base, headers=headers).json()
                    row["dimension"] = details["dimension"]
                    row["embedding_tokens_billed"] = details["tokens_billed"]
                    row["search_modes"] = {}
                    for mode in ("dense", "bm25", "hybrid"):
                        r = client.post(base + "/search", headers=headers, json={"query": "Who reviews transaction ZX-4471?", "mode": mode})
                        assert r.status_code == 200 and r.json()["results"], "retrieval failed"
                        assert "Alice" in r.json()["results"][0]["text"]
                        row["search_modes"][mode] = "passed"
                    r = client.post(base + "/answer", headers=headers, json={"question": "Who is the designated reviewer for transaction ZX-4471?", "mode": "hybrid", "max_output_tokens": 200})
                    assert r.status_code == 200, "answer request failed"
                    answer = r.json()
                    assert "Alice" in answer["answer"] and answer["citations"]["all_resolve"], "answer/citation check failed"
                    row.update(status="passed", answer=answer["answer"], model=answer.get("model"), usage=answer.get("usage"), citations_resolve=True)
                except Exception as e:
                    row["error_type"] = type(e).__name__
                finally:
                    if idx:
                        client.delete(f"/indexes/{idx['id']}", headers=headers)
                print(name, row["status"], flush=True)
    finally:
        jobs.shutdown()
        from studio.vectorstores import qdrant_store
        qdrant_store.reset_client()
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if all(r["status"] == "passed" for r in report["stores"]) and len(report["stores"]) == 5 else 1


if __name__ == "__main__":
    raise SystemExit(run())
