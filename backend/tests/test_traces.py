from studio.generation import service as gen
from test_generation import FakeChat, ask, configured, docs  # noqa: F401
from test_indexes import build, search


def test_searches_answers_and_jobs_appear_with_spans(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    search(client, auth, idx, "Which rocket carried Apollo?", mode="hybrid", top_k=3)
    monkeypatch.setattr(gen, "chat_complete", FakeChat())
    assert ask(client, auth, idx, top_k=3).status_code == 200
    t = client.get(f"/projects/{project['id']}/traces", headers=auth).json()
    kinds = {i["kind"] for i in t["items"]}
    assert kinds == {"search", "answer", "job"}
    ans = next(i for i in t["items"] if i["kind"] == "answer")
    assert {s["name"] for s in ans["spans"]} >= {"Retrieval", "Language model"} and ans["attrs"]["prompt_tokens"] == 321
    assert t["summary"]["answers"] == 1 and t["summary"]["tokens"] == 333
    only = client.get(f"/projects/{project['id']}/traces?kind=search", headers=auth).json()
    assert {i["kind"] for i in only["items"]} == {"search"}
    assert client.get("/projects/nope/traces", headers=auth).status_code == 404
