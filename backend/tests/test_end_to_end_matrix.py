"""Finite compatibility matrix. Real stores; synthetic documents; model-dependent steps are simulated.

Run: python -m pytest tests/test_end_to_end_matrix.py -q
Optional: STUDIO_MATRIX_REPORT=<path> writes machine-readable evidence, including skips.
This validates integration contracts, not answer quality on the handbook or hosted-service availability.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import upload, wait_for_job
from studio.config import settings
from studio.embeddings.providers import PROVIDERS, HashEmbedder
from studio.generation import service as gen
from studio.registry import registry
from studio.vectorstores import STORES
from test_indexes import build
from test_evaluation import dataset, add, run

METHODS = [m.id for m in registry.list("chunking") if m.status != "unavailable"]
ROWS = []
REAL_CHAT = gen.chat_complete
QUERY = "ZX-4471"
TEXT = "Transaction ZX-4471 requires approval before deployment."


@pytest.fixture(scope="module", autouse=True)
def report():
    yield
    target = os.getenv("STUDIO_MATRIX_REPORT")
    if target:
        p = Path(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"scope": "Synthetic API compatibility, real databases; simulated model tasks explicitly labelled", "application_database": "postgresql" if os.getenv("STUDIO_TEST_DATABASE_URL") else "sqlite", "rows": ROWS}, indent=2), encoding="utf-8")


@pytest.fixture()
def matrix_setup(client, monkeypatch):
    from studio.vectorstores import chroma_store, lancedb_store, pgvector_store
    monkeypatch.setattr(chroma_store, "_client", None)
    monkeypatch.setattr(lancedb_store, "_db", None)
    monkeypatch.setattr(lancedb_store, "_meta", {})
    monkeypatch.setattr(pgvector_store, "_engine", None)
    monkeypatch.setattr(settings, "pgvector_url", os.getenv("TEST_PGVECTOR_URL", "postgresql://postgres:studio@localhost:5434/vec"))
    monkeypatch.setattr(settings, "azure_endpoint", "https://example.invalid")
    monkeypatch.setattr(settings, "azure_api_key", "test-key")
    monkeypatch.setattr(settings, "azure_chat_deployment", "simulated-chat")

    def chat(model, messages, temperature, max_output_tokens):
        system = messages[0]["content"]
        if "propositions" in system:
            text = json.dumps({"propositions": [TEXT]})
        elif "JSON array" in system:
            import re
            n = int(re.search(r"array of (\d+)", system).group(1))
            text = json.dumps([8] * n)
        elif "situating" in system:
            text = "This passage describes deployment approval."
        elif "Rewrite" in system or "search queries" in system or "sub-questions" in system:
            text = QUERY
        elif "short passage" in system:
            text = TEXT
        else:
            text = "Transaction ZX-4471 requires approval [1]."
        return gen.ChatResult(text, 10, 5, 0, "stop", model)

    monkeypatch.setattr(gen, "chat_complete", chat)
    from studio.retrieval import rerank_local
    monkeypatch.setattr(rerank_local, "status", lambda: (True, "simulated scores"))
    class Reranker:
        def rerank(self, question, passages):
            return list(range(len(passages), 0, -1))
    monkeypatch.setattr(rerank_local, "_model", Reranker)


def source(method):
    if method == "code_aware":
        return "approval.py", ('def approve():\n    """' + TEXT + '"""\n    return True\n\ndef revise():\n    return False\n').encode()
    if method == "json_xml_structure":
        return "approval.json", json.dumps([{"rule": TEXT}, {"rule": "Unrelated gardening guidance."}]).encode()
    if method in ("html_structure", "table_rows", "structure_aware", "adaptive"):
        return "approval.html", ('<h1>Deployment</h1><p>' + TEXT + '</p><table><tr><th>Transaction</th><th>Rule</th></tr><tr><td>ZX-4471</td><td>Approval before deployment</td></tr></table><h2>Other</h2><p>Gardening uses sunlight and water.</p>').encode()
    if method in ("layout_aware", "page_aware"):
        import pymupdf
        d = pymupdf.open()
        p = d.new_page()
        p.insert_text((60, 60), "Deployment", fontsize=20)
        p.insert_text((60, 100), TEXT, fontsize=11)
        p = d.new_page()
        p.insert_text((60, 100), "Gardening uses sunlight and water.", fontsize=11)
        return "approval.pdf", d.tobytes()
    return "approval.md", ("# Deployment\n\n" + TEXT + "\n\n" + ("Reviewers verify deployment requirements and record the approval decision. " * 5) + "\n\n# Gardening\n\nPlants require sunlight and water. " * 3).encode()


def check_store(store, row):
    ok, why = STORES[store].status()
    if not ok:
        row.update(status="skipped", reason=why)
        pytest.skip(why)


@pytest.mark.parametrize("store", list(STORES))
@pytest.mark.parametrize("method", METHODS)
def test_chunker_store_task_matrix(client, auth, project, monkeypatch, matrix_setup, store, method):
    row = {"kind": "chunker-store", "store": store, "method": method, "status": "failed", "tasks": [], "models": "simulated"}
    ROWS.append(row)
    check_store(store, row)
    name, data = source(method)
    doc = upload(client, auth, project["id"], name, data).json()["results"][0]
    assert wait_for_job(client, auth, doc["job_id"])["status"] == "succeeded"
    params = {"provider": "local_hash"} if method == "semantic" else {}
    idx = build(client, auth, project, chunking={"method": method, "params": params}, store=store)
    base = f"/indexes/{idx['id']}"
    row["tasks"].append("ingest-build")
    chunks = client.get(base + "/chunks", headers=auth).json()["items"]
    assert chunks and idx["stored_vectors"] == idx["chunk_count"]
    assert client.get(base + f"/chunks/{chunks[0]['id']}/vector", headers=auth).status_code == 200
    row["tasks"].append("stored-vector")
    for mode in ("dense", "bm25", "hybrid"):
        for expansion in ("none", "parent", "neighbors"):
            r = client.post(base + "/search", headers=auth, json={"query": QUERY, "mode": mode, "expand": expansion, "top_k": 3})
            assert r.status_code == 200, r.text
            assert r.json()["results"] and all(h["citation"] for h in r.json()["results"])
            assert any(QUERY in h["text"] for h in r.json()["results"])
            row["tasks"].append(f"{mode}:{expansion}")
    r = client.post(base + "/search", headers=auth, json={"query": QUERY, "document_ids": ["missing"]})
    assert r.status_code == 200 and r.json()["results"] == []
    r = client.post(base + "/search", headers=auth, json={"query": QUERY, "document_ids": [doc["document_id"]], "types": ["text", "table"]})
    assert r.status_code == 200 and r.json()["results"]
    row["tasks"].append("filters")
    for transform in ("none", "rewrite", "multi_query", "hyde", "decompose"):
        for reranker in ("none", "llm", "cross_encoder"):
            r = client.post(base + "/retrieve", headers=auth, json={"query": QUERY, "mode": "hybrid", "transform": transform, "rerank": reranker, "mmr": True, "expand": "neighbors"})
            assert r.status_code == 200, r.text
            assert r.json()["results"]
            row["tasks"].append(f"pipeline:{transform}:{reranker}:mmr")
    r = client.post(base + "/answer", headers=auth, json={"question": QUERY, "mode": "hybrid", "expand": "neighbors"})
    assert r.status_code == 200, r.text
    assert r.json()["citations"]["all_resolve"]
    row["tasks"].append("answer-citations-simulated")
    ds = dataset(client, auth, project)
    add(client, auth, ds, QUERY, doc["document_id"], keywords=["ZX-4471"])
    evaluation = run(client, auth, ds, idx)
    assert evaluation["status"] == "done", evaluation
    assert evaluation["metrics"]["failed_questions"] == 0, evaluation
    assert evaluation["metrics"]["retrieval"]["hit"]["mean"] == 1.0, evaluation
    row["tasks"].append("evaluation")
    assert client.delete(base, headers=auth).status_code == 204
    assert STORES[store].count(idx["collection"]) == 0
    row["tasks"].append("delete")
    row["status"] = "passed"


@pytest.mark.parametrize("store", list(STORES))
@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_embedding_store_wiring(client, auth, project, monkeypatch, matrix_setup, store, provider):
    row = {"kind": "embedding-store", "store": store, "provider": provider, "status": "failed", "models": "real hash; other embeddings simulated"}
    ROWS.append(row)
    check_store(store, row)
    p = PROVIDERS[provider]
    monkeypatch.setattr(p, "status", lambda: (True, "simulated embedding transport"))
    if provider != "local_hash":
        monkeypatch.setattr(p, "embed", lambda texts, **kw: HashEmbedder().embed(texts))
    doc = upload(client, auth, project["id"], "rule.txt", TEXT.encode()).json()["results"][0]
    assert wait_for_job(client, auth, doc["job_id"])["status"] == "succeeded"
    idx = build(client, auth, project, provider=provider, store=store)
    for mode in ("dense", "bm25", "hybrid"):
        r = client.post(f"/indexes/{idx['id']}/search", headers=auth, json={"query": QUERY, "mode": mode})
        assert r.status_code == 200 and r.json()["results"], r.text
    assert client.delete(f"/indexes/{idx['id']}", headers=auth).status_code == 204
    row["status"] = "passed"


@pytest.mark.parametrize("store", list(STORES))
@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "ollama"])
def test_chat_provider_store_end_to_end(client, auth, project, monkeypatch, matrix_setup, store, provider):
    import httpx
    from studio.generation import providers as chat_providers
    row = {"kind": "chat-store", "store": store, "provider": provider, "status": "failed", "models": "simulated API transport; real adapter"}
    ROWS.append(row)
    check_store(store, row)
    monkeypatch.setattr(gen, "chat_complete", REAL_CHAT)
    for key in ("azure_endpoint", "azure_api_key", "azure_chat_deployment", "openai_api_key", "anthropic_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, key, "")
    if provider != "ollama":
        monkeypatch.setattr(settings, provider + "_api_key", "test-key")
    original_status = chat_providers.status
    monkeypatch.setattr(chat_providers, "status", lambda p: (True, "simulated server") if p == "ollama" and provider == "ollama" else ((False, "disabled") if p == "ollama" else original_status(p)))
    calls = []
    def post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        text = "Transaction ZX-4471 requires approval [1]."
        if "anthropic" in url:
            body = {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": 10, "output_tokens": 5}, "stop_reason": "end_turn"}
        elif "generativelanguage" in url:
            body = {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}
        elif "/api/chat" in url:
            body = {"message": {"content": text}, "prompt_eval_count": 10, "eval_count": 5, "done_reason": "stop"}
        else:
            body = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return httpx.Response(200, json=body)
    monkeypatch.setattr(httpx, "post", post)
    meta = client.get("/providers", headers=auth).json()["generation"]
    assert meta["configured"] and meta["default_model"].startswith(provider + ":")
    doc = upload(client, auth, project["id"], "rule.txt", TEXT.encode()).json()["results"][0]
    assert wait_for_job(client, auth, doc["job_id"])["status"] == "succeeded"
    idx = build(client, auth, project, store=store, contextual=True)
    r = client.post(f"/indexes/{idx['id']}/answer", headers=auth, json={"question": QUERY})
    assert r.status_code == 200, r.text
    assert r.json()["citations"]["all_resolve"] and calls
    assert client.delete(f"/indexes/{idx['id']}", headers=auth).status_code == 204
    row["status"] = "passed"
