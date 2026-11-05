"""Recommendations: every stage is covered, picks follow the documents, and measured results override rules."""
import json
from types import SimpleNamespace

from conftest import upload, wait_for_job
from studio.recommend import _clearly_better
from test_parsers_alt import scanned_pdf_and_png

STAGES = ["parsing", "chunking", "embedding", "vector_store", "retrieval", "reranking", "generation", "evaluation"]


def rec(client, auth, project):
    r = client.get(f"/projects/{project['id']}/recommendations", headers=auth)
    assert r.status_code == 200, r.text
    return r.json()


def stage(r, sid):
    return next(s for s in r["stages"] if s["id"] == sid)


def ingest(client, auth, project, name, data, **form):
    res = upload(client, auth, project["id"], name, data, **form).json()["results"][0]
    assert wait_for_job(client, auth, res["job_id"], timeout=120)["status"] == "succeeded"
    return res["document_id"]


def test_empty_project_gets_safe_defaults_for_every_stage(client, auth, project):
    r = rec(client, auth, project)
    assert [s["id"] for s in r["stages"]] == STAGES
    assert all(s["reasons"] and s["basis"] in ("measured", "rule", "default") for s in r["stages"])
    assert stage(r, "chunking")["pick"]["method"] == "recursive" and stage(r, "chunking")["basis"] == "default"
    assert stage(r, "evaluation")["action"]["href"] == "/evaluation"


def test_picks_follow_the_documents(client, auth, project):
    code = "import os\n\n\ndef a():\n    return 1\n\n\ndef b():\n    return 2\n"
    doc = ingest(client, auth, project, "tool.py", (code * 3).encode())
    one = client.get(f"/documents/{doc}/recommendation", headers=auth).json()
    assert one["chunking"]["method"] == "code_aware" and "source code" in one["chunking"]["why"] and one["profile"]["code"] is True
    data = json.dumps([{"id": i, "name": f"n{i}"} for i in range(30)], indent=2).encode()
    jd = ingest(client, auth, project, "records.json", data)
    assert client.get(f"/documents/{jd}/recommendation", headers=auth).json()["chunking"]["method"] == "json_xml_structure"
    r = rec(client, auth, project)
    ch = stage(r, "chunking")
    assert ch["basis"] == "rule" and ch["pick"]["method"] in ("code_aware", "json_xml_structure") and len(ch["reasons"]) >= 3
    assert any("tool.py" in e["label"] for e in ch["evidence"])
    assert r["documents"] == 2 and r["tokens"] > 0


def test_scanned_pdf_is_sent_to_ocr(client, auth, project):
    pdf, _ = scanned_pdf_and_png()
    ingest(client, auth, project, "scan.pdf", pdf)
    p = stage(rec(client, auth, project), "parsing")
    assert p["pick"]["method"] == "rapidocr" and "scanned" in " ".join(p["reasons"])


def test_embedding_falls_back_to_local_semantic_then_demo(client, auth, project):
    e = stage(rec(client, auth, project), "embedding")
    assert e["pick"]["method"] == "fastembed" and "no key" in " ".join(e["reasons"]).lower()


def run(mean, ci, n):
    return SimpleNamespace(metrics={"answerable": n, "retrieval": {"ndcg": {"mean": mean, "ci95": ci}}})


def test_a_measured_difference_needs_enough_questions_and_a_clear_gap():
    assert _clearly_better(run(0.80, [0.70, 0.88], 30), run(0.60, [0.5, 0.7], 30))          # interval clears the other mean
    assert not _clearly_better(run(0.80, [0.55, 0.9], 30), run(0.70, [0.5, 0.8], 30))       # overlap: noise
    assert not _clearly_better(run(0.90, [0.8, 0.95], 5), run(0.50, [0.4, 0.6], 5))          # too few questions
    assert _clearly_better(run(0.80, None, 12), run(0.70, None, 12))                         # no interval: 0.03 rule
    assert not _clearly_better(run(0.71, None, 12), run(0.70, None, 12))
