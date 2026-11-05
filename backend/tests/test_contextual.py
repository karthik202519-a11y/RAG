"""Contextual embedding: the model-written sentence is stored on the chunk and changes what gets embedded."""
from studio.generation import service as gen
from studio.jobs.index_job import embed_input
from test_generation import FakeChat, configured, docs  # noqa: F401
from test_indexes import build


def test_embed_input_prepends_context_only_when_present():
    assert embed_input("body", "text", "Sec", True) == "Section: Sec\nbody"
    assert embed_input("body", "text", "Sec", True, "About X.") == "About X.\nSection: Sec\nbody"
    assert embed_input("body", "table", "Sec", True, None) == "body"


def test_contextual_index_stores_the_sentence(client, auth, project, docs, configured, monkeypatch):
    monkeypatch.setattr(gen, "chat_complete", FakeChat("This passage is about the Apollo programme."))
    idx = build(client, auth, project, contextual=True)
    rows = client.get(f"/indexes/{idx['id']}/chunks", headers=auth).json()
    items = rows["items"] if isinstance(rows, dict) else rows
    assert items and all(c["context"] == "This passage is about the Apollo programme." for c in items if c.get("context") is not None)
    assert any(c["context"] for c in items)


def test_contextual_needs_the_chat_model(client, auth, project, docs):
    r = client.post(f"/projects/{project['id']}/indexes", headers=auth,
                    json={"name": "x", "provider": "local_hash", "contextual": True,
                          "chunking": {"method": "recursive", "params": {}}})
    assert r.status_code == 409 and "chat model" in r.json()["detail"]
