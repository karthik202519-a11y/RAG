"""Multi-stage retrieval: query transforms, MMR, LLM reranking (fake chat model; tests never call Azure)."""
import pytest

from conftest import upload, wait_for_job
from studio.config import settings
from studio.generation import service as gen
from test_indexes import CHUNKING, build, make_docs  # noqa: F401


@pytest.fixture()
def docs(client, auth, project):
    return make_docs(client, auth, project)


@pytest.fixture()
def configured(monkeypatch):
    monkeypatch.setattr(settings, "azure_endpoint", "https://example.invalid")
    monkeypatch.setattr(settings, "azure_api_key", "not-a-real-key")
    monkeypatch.setattr(settings, "azure_chat_deployment", "fake-chat")


class Chat:
    """Scripted chat model: maps a keyword in the system prompt to a reply."""
    def __init__(self, **replies):
        self.replies, self.calls = replies, []

    def __call__(self, model, messages, temperature, max_output_tokens):
        sys, user = messages[0]["content"], messages[1]["content"]
        self.calls.append((sys, user))
        for key, reply in self.replies.items():
            if key.replace("_", " ") in sys.lower().replace("-", " "):
                text = reply(user) if callable(reply) else reply
                return gen.ChatResult(text, 100, 20, 0, "stop", model)
        raise AssertionError(f"unscripted prompt: {sys[:80]}")


def retrieve(client, auth, idx, **body):
    r = client.post(f"/indexes/{idx['id']}/retrieve", headers=auth, json={"query": "rocket for the moon", **body})
    return r


def stage(r, name):
    return next(s for s in r["stages"] if s["stage"] == name)


def test_plain_pipeline_matches_base_search_and_reports_stages(client, auth, project, docs):
    idx = build(client, auth, project)
    r = retrieve(client, auth, idx, query="Saturn V rocket Apollo lunar orbit").json()
    base = client.post(f"/indexes/{idx['id']}/search", headers=auth, json={"query": "Saturn V rocket Apollo lunar orbit"}).json()
    assert [h["chunk_id"] for h in r["results"]] == [h["chunk_id"] for h in base["results"]]
    assert [s["stage"] for s in r["stages"]] == ["query_transform", "retrieve", "fuse"]
    assert r["llm_tokens"] == 0 and r["results"][0]["final_rank"] == 1


def test_rewrite_adds_a_rewritten_query_and_finds_it(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    chat = Chat(**{"standalone": "Saturn V Apollo spacecraft lunar orbit"})
    monkeypatch.setattr(gen, "chat_complete", chat)
    r = retrieve(client, auth, idx, query="how did they get up there", transform="rewrite", mode="bm25").json()
    qs = stage(r, "query_transform")["queries"]
    assert [q["role"] for q in qs] == ["original", "rewrite"] and qs[1]["text"].startswith("Saturn V")
    assert [q["search"] for q in qs] == [False, True]  # the rewrite replaces the original, which stays visible
    assert [q["role"] for q in stage(r, "retrieve")["per_query"]] == ["rewrite"]
    assert r["results"][0]["filename"] == "apollo.txt"
    assert stage(r, "query_transform")["llm_tokens"] == 120 and r["llm_tokens"] == 120


def test_multi_query_fuses_variants(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    monkeypatch.setattr(gen, "chat_complete", Chat(**{"different search queries": "1. Saturn V rocket\n2. Apollo 11 landing Tranquility\n- lunar orbit spacecraft"}))
    r = retrieve(client, auth, idx, query="moon missions", transform="multi_query", n_queries=3, mode="bm25").json()
    qs = stage(r, "query_transform")["queries"]
    assert len(qs) == 4 and [q["role"] for q in qs[1:]] == ["variant"] * 3  # numbering and bullets are stripped
    assert qs[1]["text"] == "Saturn V rocket"
    assert stage(r, "fuse")["n_queries"] == 4
    assert all(h["queries_found_by"] >= 1 for h in r["results"])
    assert any(h["queries_found_by"] > 1 for h in r["results"])  # a chunk matched by several variants rises


def test_hyde_embeds_the_hypothetical_passage_but_not_for_keyword_search(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    monkeypatch.setattr(gen, "chat_complete", Chat(**{"short passage": "The Saturn V rocket carried the Apollo spacecraft toward lunar orbit."}))
    r = retrieve(client, auth, idx, query="what launched astronauts", transform="hyde", mode="dense").json()
    roles = [q["role"] for q in stage(r, "query_transform")["queries"]]
    assert roles == ["original", "hypothetical_passage"] and r["results"][0]["filename"] == "apollo.txt"
    assert [q["role"] for q in stage(r, "retrieve")["per_query"]] == ["hypothetical_passage"]  # embedded instead of the question
    r2 = retrieve(client, auth, idx, query="what launched astronauts", transform="hyde", mode="bm25").json()
    assert [q["role"] for q in stage(r2, "retrieve")["per_query"]] == ["original"]  # never used for keyword matching
    r3 = retrieve(client, auth, idx, query="what launched astronauts", transform="hyde", mode="hybrid").json()
    pq = stage(r3, "retrieve")["per_query"]
    assert [(q["role"], q["mode"]) for q in pq] == [("original", "bm25"), ("hypothetical_passage", "dense")]


def test_decompose_searches_each_subquestion(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    monkeypatch.setattr(gen, "chat_complete", Chat(**{"sub questions": "Which rocket carried Apollo?\nWhat does chlorophyll absorb?"}))
    r = retrieve(client, auth, idx, query="rocket and chlorophyll", transform="decompose", mode="bm25", top_k=6).json()
    assert [q["role"] for q in stage(r, "query_transform")["queries"]] == ["original", "sub_question", "sub_question"]
    names = {h["filename"] for h in r["results"]}
    assert {"apollo.txt", "plants.txt"} <= names


def test_llm_features_need_the_chat_model(client, auth, project, docs):
    idx = build(client, auth, project)
    for body in ({"transform": "rewrite"}, {"rerank": "llm"}):
        r = retrieve(client, auth, idx, **body)
        assert r.status_code == 409 and "chat model" in r.json()["detail"]
    assert retrieve(client, auth, idx, mmr=True).status_code == 200  # MMR needs no LLM


def test_llm_rerank_reorders_and_reports_every_move(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    base = retrieve(client, auth, idx, query="light", top_k=4, mode="dense").json()["results"]

    def reverse_scores(user):
        n = user.count("\n[")  # passages are numbered [1]..[n]
        n = max(n, 1)
        return str(list(range(0, n)))  # passage 1 gets 0, the last gets the highest

    monkeypatch.setattr(gen, "chat_complete", Chat(**{"rate how well": reverse_scores}))
    r = retrieve(client, auth, idx, query="light", top_k=4, mode="dense", rerank="llm", rerank_candidates=4).json()
    rr = stage(r, "rerank")
    assert rr["method"] == "llm" and rr["pool"] == 4 and rr["warning"] is None
    ids = [h["chunk_id"] for h in r["results"]]
    assert ids == [h["chunk_id"] for h in reversed(base[:4])]
    top = r["results"][0]
    assert top["rerank"]["before_rank"] == 4 and top["rerank"]["after_rank"] == 1 and top["rerank"]["moved"] == 3
    assert "dense" in top["contributions"]  # the original retrieval scores are kept next to the reranker's


def test_rerank_failure_keeps_original_order_and_says_so(client, auth, project, docs, configured, monkeypatch):
    idx = build(client, auth, project)
    base = retrieve(client, auth, idx, query="light", top_k=3).json()["results"]
    monkeypatch.setattr(gen, "chat_complete", Chat(**{"rate how well": "I think passage two is best."}))
    r = retrieve(client, auth, idx, query="light", top_k=3, rerank="llm", rerank_candidates=3).json()
    assert [h["chunk_id"] for h in r["results"]] == [h["chunk_id"] for h in base]
    assert r["warnings"] and "original order kept" in r["warnings"][0]
    monkeypatch.setattr(gen, "chat_complete", Chat(**{"rate how well": "[5, 5]"}))  # wrong number of scores
    r = retrieve(client, auth, idx, query="light", top_k=3, rerank="llm", rerank_candidates=3).json()
    assert [h["chunk_id"] for h in r["results"]] == [h["chunk_id"] for h in base] and r["warnings"]


def test_mmr_demotes_near_duplicates(client, auth, project):
    for n, body in {"a.txt": "Volcanic magma rises through cracks in the crust and erupts as lava.",
                    "b.txt": "Volcanic magma rises through cracks in the crust and erupts as lava today.",
                    "c.txt": "Glaciers are rivers of ice that flow slowly down mountain valleys."}.items():
        j = upload(client, auth, project["id"], n, body.encode()).json()["results"][0]
        wait_for_job(client, auth, j["job_id"])
    idx = build(client, auth, project, chunking={"method": "page_aware", "params": {"max_tokens": 0}})
    plain = retrieve(client, auth, idx, query="volcanic magma lava crust", top_k=2).json()["results"]
    assert {h["filename"] for h in plain} == {"a.txt", "b.txt"}  # two near-identical chunks win
    div = retrieve(client, auth, idx, query="volcanic magma lava crust", top_k=2, mmr=True, mmr_lambda=0.2).json()
    assert "c.txt" in {h["filename"] for h in div["results"]}
    assert stage(div, "mmr")["changed_top_k"] is True
    assert div["results"][1]["mmr"]["max_similarity_to_selected"] < 0.9  # the second pick is not a near-copy of the first


def test_pipeline_validation_and_access(client, auth, project, docs):
    from conftest import register
    idx = build(client, auth, project)
    assert retrieve(client, auth, idx, transform="bogus").status_code == 422
    assert retrieve(client, auth, idx, mmr_lambda=2).status_code == 422
    assert retrieve(client, auth, idx, n_queries=0).status_code == 422
    eve = register(client, "eve@example.com", "Eve")
    assert retrieve(client, eve, idx).status_code == 404
    assert client.post(f"/indexes/{idx['id']}/retrieve", json={"query": "x"}).status_code == 401
