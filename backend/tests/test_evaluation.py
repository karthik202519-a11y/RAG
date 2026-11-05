"""Evaluation datasets, runs and comparison through the API (offline hash embedder)."""
import pytest

from conftest import register
from studio.config import settings
from studio.generation import service as gen
from test_indexes import CHUNKING, build, make_docs  # noqa: F401
from conftest import wait_for_job


@pytest.fixture()
def docs(client, auth, project):
    return make_docs(client, auth, project)


@pytest.fixture()
def idx(client, auth, project, docs):
    return build(client, auth, project)


def dataset(client, auth, project, name="qa"):
    r = client.post(f"/projects/{project['id']}/eval-datasets", headers=auth, json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


def add(client, auth, ds, question, doc=None, pages=(1,), **kw):
    body = {"question": question, "evidence": [{"document": doc, "pages": list(pages)}] if doc else [], **kw}
    r = client.post(f"/eval-datasets/{ds['id']}/questions", headers=auth, json=body)
    assert r.status_code == 201, r.text
    return r.json()


def run(client, auth, ds, idx, timeout=60, **cfg):
    r = client.post(f"/eval-datasets/{ds['id']}/runs", headers=auth, json={"index_id": idx["id"], **cfg})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        x = client.get(f"/eval-runs/{rid}", headers=auth).json()
        if x["status"] in ("done", "failed"):
            return x
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def seed_questions(client, auth, ds):
    add(client, auth, ds, "Which rocket carried the Apollo spacecraft toward lunar orbit?", "apollo", qtype="prose", keywords=["Saturn V"])
    add(client, auth, ds, "What absorbs red and blue light and reflects green light?", "plants", qtype="prose", keywords=["Chlorophyll"])
    add(client, auth, ds, "What does an index speed up at the cost of extra storage?", "database", qtype="prose")
    add(client, auth, ds, "How many moons does Jupiter have?", None, answerable=False, qtype="unanswerable")


def test_question_validation(client, auth, project):
    ds = dataset(client, auth, project)
    r = client.post(f"/eval-datasets/{ds['id']}/questions", headers=auth, json={"question": "No evidence given"})
    assert r.status_code == 422 and "evidence" in r.json()["detail"]
    assert client.post(f"/eval-datasets/{ds['id']}/questions", headers=auth, json={"question": "", "answerable": False}).status_code == 422
    q = add(client, auth, ds, "Fine", None, answerable=False)
    assert q["evidence"] == [] and q["answerable"] is False
    upd = client.patch(f"/eval-questions/{q['id']}", headers=auth, json={"question": "Better", "answerable": True,
                                                                         "evidence": [{"document": "x", "pages": [1]}]})
    assert upd.status_code == 200 and upd.json()["question"] == "Better"
    assert client.delete(f"/eval-questions/{q['id']}", headers=auth).status_code == 204
    assert client.get(f"/eval-datasets/{ds['id']}", headers=auth).json()["questions"] == []


def test_run_computes_real_metrics_and_flags_unanswerable(client, auth, project, idx):
    ds = dataset(client, auth, project)
    seed_questions(client, auth, ds)
    r = run(client, auth, ds, idx, mode="dense", k=3)
    assert r["status"] == "done" and r["metrics"]["questions"] == 4 and r["metrics"]["answerable"] == 3 and r["metrics"]["unanswerable"] == 1
    m = r["metrics"]["retrieval"]
    assert m["hit"]["n"] == 3 and m["hit"]["mean"] == 1.0 and m["mrr"]["mean"] > 0.5  # the hash embedder is good at word overlap
    assert m["hit"]["ci95"] is None  # 3 questions are too few for an interval
    assert m["keyword_coverage"]["n"] == 2
    un = [p for p in r["per_question"] if not p["answerable"]][0]
    assert un["metrics"]["hit"] is None and un["top1_dense_score"] is not None
    gate = r["metrics"]["unanswerable_gate"]
    assert gate["top1_dense_unanswerable"] and gate["sweep"]
    assert r["metrics"]["latency_ms"]["p50"] is not None and "answers" in r["metrics"] and r["metrics"]["answers"] is None


def test_wrong_labels_produce_a_miss_not_a_crash(client, auth, project, idx):
    ds = dataset(client, auth, project)
    add(client, auth, ds, "Which rocket carried the Apollo spacecraft?", "apollo", pages=(7,))  # wrong page
    r = run(client, auth, ds, idx, k=5)
    p = r["per_question"][0]["metrics"]
    assert p["hit"] == 0.0 and p["first_rank"] is None and r["metrics"]["retrieval"]["hit"]["mean"] == 0.0


def test_compare_runs_pairs_questions_and_lists_changes(client, auth, project, idx):
    ds = dataset(client, auth, project)
    seed_questions(client, auth, ds)
    a = run(client, auth, ds, idx, mode="dense", k=3, label="dense")
    b = run(client, auth, ds, idx, mode="bm25", k=3, label="bm25")
    c = client.get(f"/eval-datasets/{ds['id']}/compare", headers=auth, params={"runs": f"{a['id']},{b['id']}"}).json()
    assert [r["label"] for r in c["runs"]] == ["dense", "bm25"]
    assert {row["metric"] for row in c["metrics"]} >= {"hit", "mrr", "ndcg", "recall", "precision", "ap"}
    d = c["paired_vs_first"][0]
    assert d["shared_questions"] == 3 and d["mrr"]["n"] == 3 and d["mrr"]["significant"] is None  # too few pairs
    assert all(ch["direction"] in ("improved", "regressed") for ch in d["changed_questions"])
    assert client.get(f"/eval-datasets/{ds['id']}/compare", headers=auth, params={"runs": a["id"]}).status_code == 422
    assert client.get(f"/eval-datasets/{ds['id']}/compare", headers=auth, params={"runs": f"{a['id']},nope"}).status_code == 404


def test_paired_difference_reports_significance_with_enough_questions():
    from studio.evaluation.service import paired_diff
    better = paired_diff([0.0] * 20, [1.0] * 20)
    assert better["mean_diff"] == 1.0 and better["significant"] is True and better["ci95"] == [1.0, 1.0]
    noise = paired_diff([0, 1] * 10, [1, 0] * 10)
    assert noise["significant"] is False and noise["ci95"][0] < 0 < noise["ci95"][1]
    assert paired_diff([0.5] * 5, [0.6] * 5)["significant"] is None


def test_import_native_prototype_formats_and_missing_documents(client, auth, project, docs):
    ds = dataset(client, auth, project)
    r = client.post(f"/eval-datasets/{ds['id']}/import", headers=auth, json={"questions": [
        {"question": "Native one", "evidence": [{"document": "apollo", "pages": [1]}], "keywords": ["Saturn"], "type": "prose"}]}).json()
    assert r["format"] == "native" and r["added"] == 1
    hard = [{"id": "N1", "cat": "multi-hop", "doc": "apollo", "question": "Two groups", "page_groups": [[1], [2]], "keywords": []},
            {"id": "I1", "cat": "number", "doc": "ipcc", "question": "About climate", "page_groups": [[3]], "keywords": []},
            {"id": "U1", "cat": "unanswerable", "doc": "nasa", "question": "Nobody knows", "page_groups": [], "keywords": []}]
    r = client.post(f"/eval-datasets/{ds['id']}/import", headers=auth, json={"questions": hard}).json()
    assert r["format"] == "prototype_hard" and r["added"] == 2 and r["skipped"][0]["reason"].startswith("document not in this project")
    basic = [{"id": 1, "question": "Basic", "type": "table", "pdf_pages": [1, 2]}]
    r = client.post(f"/eval-datasets/{ds['id']}/import", headers=auth, json={"questions": basic, "default_document": "plants"}).json()
    assert r["format"] == "prototype_basic" and r["added"] == 1
    d = client.get(f"/eval-datasets/{ds['id']}", headers=auth).json()
    assert len(d["questions"]) == 4 and d["unanswerable"] == 1
    multi = next(q for q in d["questions"] if q["question"] == "Two groups")
    assert len(multi["evidence"]) == 2  # multi-hop keeps both evidence groups
    assert client.post(f"/eval-datasets/{ds['id']}/import", headers=auth, json={"questions": [{"no": "question"}]}).status_code == 422
    assert client.post(f"/eval-datasets/{ds['id']}/import", headers=auth, json={"questions": [], }).status_code == 422


def test_splits_filter_runs_and_warn_about_test_reuse(client, auth, project, idx):
    ds = dataset(client, auth, project)
    for i in range(10):
        add(client, auth, ds, f"Which rocket carried the Apollo spacecraft variant {i}?", "apollo")
    counts = client.post(f"/eval-datasets/{ds['id']}/assign-splits", headers=auth, json={"train": 0.5, "validation": 0.2, "seed": 4}).json()
    assert counts == {"train": 5, "validation": 2, "test": 3}
    assert client.post(f"/eval-datasets/{ds['id']}/assign-splits", headers=auth, json={"train": 0.8, "validation": 0.4}).status_code == 422
    r = run(client, auth, ds, idx, split="validation", k=3)
    assert r["metrics"]["questions"] == 2 and r["config"]["split"] == "validation"
    run(client, auth, ds, idx, split="test", k=3)
    d = client.get(f"/eval-datasets/{ds['id']}", headers=auth).json()
    assert d["runs_on_test_split"] == 1 and "test split once" in d["guidance"]
    full = run(client, auth, ds, idx, split="all", k=3)
    assert full["metrics"]["retrieval"]["hit"]["n"] == 10 and full["metrics"]["retrieval"]["hit"]["ci95"] is not None  # 10 is enough for a CI
    assert client.post(f"/eval-datasets/{ds['id']}/runs", headers=auth, json={"index_id": idx["id"], "split": "bogus"}).status_code == 422


def test_answer_level_checks_without_pretending_to_judge_quality(client, auth, project, idx, monkeypatch):
    monkeypatch.setattr(settings, "azure_endpoint", "https://example.invalid")
    monkeypatch.setattr(settings, "azure_api_key", "not-a-real-key")
    monkeypatch.setattr(settings, "azure_chat_deployment", "fake-chat")

    def fake(model, messages, temperature, max_output_tokens):
        q = messages[1]["content"].lower()
        text = "I cannot answer this from the provided documents." if "jupiter" in q else "It was the Saturn V [1]."
        return gen.ChatResult(text, 200, 10, 0, "stop", model)

    monkeypatch.setattr(gen, "chat_complete", fake)
    ds = dataset(client, auth, project)
    seed_questions(client, auth, ds)
    r = run(client, auth, ds, idx, k=3, with_answers=True)
    a = r["metrics"]["answers"]
    assert a["n"] == 4 and a["abstention_correct"]["n"] == 4 and a["tokens"] > 0
    assert a["abstention_correct"]["rate"] == 1.0  # the unanswerable one is refused and the three answerable ones answered
    assert a["citations_resolve"]["rate"] == 1.0 and a["keyword_hit"]["rate"] is not None
    assert any("faithfulness" in x for x in a["not_available"])  # it does not claim to measure what it cannot


def test_one_failing_question_does_not_sink_the_run(client, auth, project, idx, monkeypatch):
    from studio import retrieval
    real = retrieval.search

    def flaky(db, index, query, *a, **k):
        if "Jupiter" in query:
            raise RuntimeError("boom")
        return real(db, index, query, *a, **k)

    monkeypatch.setattr(retrieval, "search", flaky)
    ds = dataset(client, auth, project)
    seed_questions(client, auth, ds)
    r = run(client, auth, ds, idx, k=3)
    assert r["status"] == "done" and r["metrics"]["failed_questions"] == 1
    assert any(p.get("error") and "boom" in p["error"] for p in r["per_question"])


def test_export_csv_and_json(client, auth, project, idx):
    ds = dataset(client, auth, project)
    seed_questions(client, auth, ds)
    r = run(client, auth, ds, idx, k=3)
    csv_ = client.get(f"/eval-runs/{r['id']}/export", headers=auth, params={"format": "csv"})
    assert csv_.status_code == 200 and csv_.text.splitlines()[0].startswith("question,type,answerable,split") and len(csv_.text.splitlines()) == 5
    js = client.get(f"/eval-runs/{r['id']}/export", headers=auth).json()
    assert js["id"] == r["id"] and len(js["per_question"]) == 4


def test_access_control_and_cleanup(client, auth, project, idx):
    ds = dataset(client, auth, project)
    seed_questions(client, auth, ds)
    r = run(client, auth, ds, idx, k=3)
    eve = register(client, "eve@example.com", "Eve")
    for path in (f"/eval-datasets/{ds['id']}", f"/eval-datasets/{ds['id']}/runs", f"/eval-runs/{r['id']}", f"/eval-runs/{r['id']}/export",
                 f"/projects/{project['id']}/eval-datasets"):
        assert client.get(path, headers=eve).status_code == 404, path
    assert client.post(f"/eval-datasets/{ds['id']}/runs", headers=eve, json={"index_id": idx["id"]}).status_code == 404
    assert client.delete(f"/eval-datasets/{ds['id']}", headers=eve).status_code == 404
    assert client.get(f"/eval-datasets/{ds['id']}").status_code == 401
    assert client.delete(f"/eval-runs/{r['id']}", headers=auth).status_code == 204
    assert client.delete(f"/eval-datasets/{ds['id']}", headers=auth).status_code == 204
    assert client.get(f"/eval-datasets/{ds['id']}", headers=auth).status_code == 404


def test_run_requires_a_ready_index_and_questions(client, auth, project, idx):
    ds = dataset(client, auth, project)
    assert client.post(f"/eval-datasets/{ds['id']}/runs", headers=auth, json={"index_id": idx["id"]}).status_code == 422  # no questions
    seed_questions(client, auth, ds)
    assert client.post(f"/eval-datasets/{ds['id']}/runs", headers=auth, json={"index_id": "0" * 32}).status_code == 404
    assert client.post(f"/eval-datasets/{ds['id']}/runs", headers=auth, json={"index_id": idx["id"], "k": 0}).status_code == 422


def test_grid_creates_one_run_per_combination_and_caps_size(client, auth, project, idx):
    ds = dataset(client, auth, project)
    seed_questions(client, auth, ds)
    g = client.post(f"/eval-datasets/{ds['id']}/grid", headers=auth,
                    json={"index_ids": [idx["id"]], "modes": ["dense", "bm25"], "ks": [3, 5]})
    assert g.status_code == 201, g.text
    assert g.json()["count"] == 4
    labels = {r["label"] for r in g.json()["runs"]}
    assert len(labels) == 4  # each combination is distinguishable
    big = client.post(f"/eval-datasets/{ds['id']}/grid", headers=auth,  # 2 x 3 x 4 x 2 = 48 combinations: refused before any run starts
                      json={"index_ids": ["a", "b"], "modes": ["dense", "bm25", "hybrid"], "ks": [1, 3, 5, 10], "rerank": ["none", "llm"]})
    assert big.status_code == 422 and "limit is 24" in big.json()["detail"]
    assert client.post(f"/eval-datasets/{ds['id']}/grid", headers=auth, json={"index_ids": ["nope"]}).status_code == 404
