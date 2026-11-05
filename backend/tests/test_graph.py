"""Knowledge graph: extraction verified against chunk text, build job, local search. The chat model is faked."""
import json
import time

from studio.generation import service as gen
from studio.graph.extract import extract_chunk
from test_generation import FakeChat, configured, docs  # noqa: F401
from test_indexes import build

REPLY = json.dumps({
    "entities": [{"name": "Saturn V", "type": "rocket"}, {"name": "Apollo spacecraft", "type": "vehicle"},
                 {"name": "Death Star", "type": "invented"}],  # not in any chunk: must be dropped
    "relations": [{"source": "Saturn V", "relation": "carried", "target": "Apollo spacecraft"},
                  {"source": "Saturn V", "relation": "destroyed", "target": "Death Star"}]})


def test_extraction_drops_anything_not_in_the_text(configured, monkeypatch):
    monkeypatch.setattr(gen, "chat_complete", FakeChat(REPLY))
    out = extract_chunk("The Saturn V rocket carried the Apollo spacecraft toward lunar orbit.", "fake-chat")
    assert {e["norm"] for e in out["entities"]} == {"saturn v", "apollo spacecraft"}
    assert [(r["source"], r["target"]) for r in out["relations"]] == [("saturn v", "apollo spacecraft")]
    assert out["dropped"] == 2  # the invented entity and the relation that depended on it


def test_build_and_search_the_graph(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    assert client.post(f"/indexes/{idx['id']}/graph/search", headers=auth, json={"query": "Saturn V"}).status_code == 409  # not built yet
    monkeypatch.setattr(gen, "chat_complete", FakeChat(REPLY))
    r = client.post(f"/indexes/{idx['id']}/graph", headers=auth, json={"max_chunks": 10, "min_tokens": 1})
    assert r.status_code == 202, r.text
    for _ in range(100):
        g = client.get(f"/indexes/{idx['id']}/graph", headers=auth).json()
        if g["build"]["status"] in ("ready", "failed"):
            break
        time.sleep(0.2)
    assert g["build"]["status"] == "ready", g
    assert g["graph"]["entities"] >= 2 and g["graph"]["relations"] >= 1
    assert all(n["name"] != "Death Star" for n in g["graph"]["nodes"])
    s = client.post(f"/indexes/{idx['id']}/graph/search", headers=auth, json={"query": "What did the Saturn V carry?"}).json()
    assert [x["name"] for x in s["seeds"]] == ["Saturn V"]
    assert s["results"] and s["results"][0]["via"][0]["relation"] == "carried"
    none = client.post(f"/indexes/{idx['id']}/graph/search", headers=auth, json={"query": "photosynthesis"}).json()
    assert none["results"] == [] and none["note"]
    assert client.delete(f"/indexes/{idx['id']}/graph", headers=auth).status_code == 204
    assert client.get(f"/indexes/{idx['id']}/graph", headers=auth).json()["build"] is None


def test_build_needs_the_chat_model(client, auth, project, docs):
    idx = build(client, auth, project)
    assert client.post(f"/indexes/{idx['id']}/graph", headers=auth, json={}).status_code == 409
