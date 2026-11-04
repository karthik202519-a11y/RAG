import io
import time

import docx

from conftest import make_pdf, upload, wait_for_job
from studio import db as dbmod
from studio.models import Job


def ingest(client, h, pid, name, data, **form):
    r = upload(client, h, pid, name, data, **form)
    assert r.status_code == 201, r.text
    res = r.json()["results"][0]
    if res["status"] == "queued":
        job = wait_for_job(client, h, res["job_id"])
        assert job["status"] == "succeeded", job
    return res


def test_pdf_ingestion_end_to_end(client, auth, project, sample_pdf):
    res = ingest(client, auth, project["id"], "science.pdf", sample_pdf)
    d = client.get(f"/documents/{res['document_id']}", headers=auth).json()
    v = d["latest_version"]
    assert v["status"] == "ready" and v["pages"] == 3 and v["parser"] == "pymupdf" and v["token_count"] > 50
    content = client.get(f"/documents/{d['id']}/content", headers=auth).json()
    assert "Photosynthesis" in content["text"] and "Apollo" in content["text"]
    page2 = client.get(f"/documents/{d['id']}/content", headers=auth, params={"page": 2}).json()["page"]
    assert page2["pdf_page"] == 2
    assert client.get(f"/documents/{d['id']}/content", headers=auth, params={"page": 9}).status_code == 404
    src = client.get(f"/documents/{d['id']}/source", headers=auth)
    assert src.status_code == 200 and src.content.startswith(b"%PDF-")


def test_text_markdown_html_csv_json_docx(client, auth, project):
    pid = project["id"]
    ingest(client, auth, pid, "a.txt", b"First paragraph.\n\nSecond paragraph.")
    ingest(client, auth, pid, "b.md", b"# Title\n\nSome *markdown* body.\n\n## Part\n\nMore text.")
    ingest(client, auth, pid, "c.html", b"<html><body><h1>Head</h1><p>Body text</p><table><tr><th>a</th><th>b</th></tr>"
                                         b"<tr><td>1</td><td>2</td></tr></table><script>evil()</script></body></html>")
    ingest(client, auth, pid, "d.csv", b"name,age\nAda,36\nBob,41\n")
    ingest(client, auth, pid, "e.json", b'{"k": [1, 2, {"x": "y"}]}')
    buf = io.BytesIO()
    d = docx.Document()
    d.add_heading("Report", 1)
    d.add_paragraph("A docx paragraph.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text, t.cell(1, 0).text, t.cell(1, 1).text = "h1", "h2", "v1", "v2"
    d.save(buf)
    ingest(client, auth, pid, "f.docx", buf.getvalue())

    items = client.get(f"/projects/{pid}/documents", headers=auth).json()["items"]
    assert {i["filename"] for i in items} == {"a.txt", "b.md", "c.html", "d.csv", "e.json", "f.docx"}
    assert all(i["latest_version"]["status"] == "ready" for i in items)
    by = {i["filename"]: i for i in items}
    html = client.get(f"/documents/{by['c.html']['id']}/content", headers=auth).json()["text"]
    assert "Body text" in html and "evil()" not in html and "| a | b |" in html
    assert "| name | age |" in client.get(f"/documents/{by['d.csv']['id']}/content", headers=auth).json()["text"]
    assert "| h1 | h2 |" in client.get(f"/documents/{by['f.docx']['id']}/content", headers=auth).json()["text"]


def test_identical_upload_is_detected_as_duplicate(client, auth, project, sample_pdf):
    first = ingest(client, auth, project["id"], "x.pdf", sample_pdf)
    second = upload(client, auth, project["id"], "renamed.pdf", sample_pdf).json()["results"][0]
    assert second["status"] == "duplicate" and second["document_id"] == first["document_id"]
    assert client.get(f"/projects/{project['id']}/documents", headers=auth).json()["total"] == 1


def test_changed_content_with_same_name_becomes_a_new_version(client, auth, project):
    pid = project["id"]
    ingest(client, auth, pid, "notes.txt", b"version one")
    r2 = ingest(client, auth, pid, "notes.txt", b"version two, now longer")
    d = client.get(f"/documents/{r2['document_id']}", headers=auth).json()
    assert d["version_count"] == 2 and [v["version"] for v in d["versions"]] == [1, 2]
    assert "version two" in client.get(f"/documents/{d['id']}/content", headers=auth).json()["text"]
    old = client.get(f"/documents/{d['id']}/content", headers=auth, params={"version_id": d["versions"][0]["id"]}).json()
    assert old["text"] == "version one"


def test_rejections(client, auth, project):
    pid = project["id"]
    r = upload(client, auth, pid, "malware.exe", b"MZ....").json()["results"][0]
    assert r["status"] == "rejected" and "Unsupported" in r["error"]
    r = upload(client, auth, pid, "fake.pdf", b"this is not a pdf").json()["results"][0]
    assert r["status"] == "rejected" and "does not look like" in r["error"]
    r = upload(client, auth, pid, "empty.txt", b"").json()["results"][0]
    assert r["status"] == "rejected"
    r = upload(client, auth, pid, "bin.txt", b"abc\x00def").json()["results"][0]
    assert r["status"] == "rejected"
    r = upload(client, auth, pid, "fake.docx", b"not a zip").json()["results"][0]
    assert r["status"] == "rejected"
    r = upload(client, auth, pid, "../../etc/passwd.txt", b"path trick").json()["results"][0]
    assert r["filename"] == "passwd.txt"  # directory parts are stripped


def test_multi_file_upload_reports_each_file(client, auth, project):
    r = client.post(f"/projects/{project['id']}/documents", headers=auth, files=[
        ("files", ("one.txt", b"first doc", "text/plain")), ("files", ("two.txt", b"second doc", "text/plain")),
        ("files", ("bad.exe", b"x", "application/octet-stream"))])
    st = [x["status"] for x in r.json()["results"]]
    assert st == ["queued", "queued", "rejected"]
    for x in r.json()["results"][:2]:
        wait_for_job(client, auth, x["job_id"])


def test_corrupt_pdf_fails_cleanly_and_can_be_retried_after_fix(client, auth, project):
    bad = b"%PDF-1.4\nthis is not really a pdf\n%%EOF"
    r = upload(client, auth, project["id"], "broken.pdf", bad).json()["results"][0]
    job = wait_for_job(client, auth, r["job_id"])
    assert job["status"] == "dead" and "Could not parse" in job["error"]
    assert job["attempts"] == 1  # a corrupt file is a permanent failure, not retried 3 times
    v = client.get(f"/documents/{r['document_id']}", headers=auth).json()["latest_version"]
    assert v["status"] == "failed" and "Could not parse" in v["error"]
    assert client.get(f"/documents/{r['document_id']}/content", headers=auth).status_code == 409


def test_transient_failures_retry_with_backoff_then_succeed(client, auth, project, monkeypatch):
    from studio.jobs import runner, tasks
    real, calls = tasks.ingest_document, {"n": 0}

    def flaky(ctx):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("storage temporarily unavailable")
        real(ctx)

    monkeypatch.setitem(runner.HANDLERS, "ingest_document", flaky)
    r = upload(client, auth, project["id"], "ok.txt", b"retry me").json()["results"][0]
    job = wait_for_job(client, auth, r["job_id"])
    assert job["status"] == "succeeded" and job["attempts"] == 3 and calls["n"] == 3


def test_exhausted_retries_mark_job_dead_and_manual_retry_works(client, auth, project, monkeypatch):
    from studio.jobs import runner, tasks
    real, state = tasks.ingest_document, {"fail": True}

    def failing(ctx):
        if state["fail"]:
            raise ConnectionError("down")
        real(ctx)

    monkeypatch.setitem(runner.HANDLERS, "ingest_document", failing)
    r = upload(client, auth, project["id"], "x.txt", b"will fail then work").json()["results"][0]
    job = wait_for_job(client, auth, r["job_id"])
    assert job["status"] == "dead" and job["attempts"] == 3 and "down" in job["error"]
    assert client.get(f"/documents/{r['document_id']}", headers=auth).json()["latest_version"]["status"] == "failed"
    state["fail"] = False
    assert client.post(f"/ingestion-jobs/{job['id']}/retry", headers=auth).status_code == 200
    assert wait_for_job(client, auth, job["id"])["status"] == "succeeded"
    assert client.get(f"/documents/{r['document_id']}", headers=auth).json()["latest_version"]["status"] == "ready"
    assert client.post(f"/ingestion-jobs/{job['id']}/retry", headers=auth).status_code == 409  # already succeeded


def test_cancel_a_running_job(client, auth, project, monkeypatch):
    from studio.jobs import runner
    started = {"flag": False}

    def slow(ctx):
        started["flag"] = True
        for _ in range(200):
            if ctx.cancelled():
                raise InterruptedError("cancelled")
            time.sleep(0.02)

    monkeypatch.setitem(runner.HANDLERS, "ingest_document", slow)
    r = upload(client, auth, project["id"], "slow.txt", b"cancel me").json()["results"][0]
    while not started["flag"]:
        time.sleep(0.01)
    assert client.post(f"/ingestion-jobs/{r['job_id']}/cancel", headers=auth).status_code == 200
    assert wait_for_job(client, auth, r["job_id"])["status"] == "cancelled"
    v = client.get(f"/documents/{r['document_id']}", headers=auth).json()["latest_version"]
    assert v["status"] == "uploaded" and "cancelled" in v["error"]  # not left looking like it is still processing


def test_jobs_are_idempotent(client, auth, project):
    r = upload(client, auth, project["id"], "once.txt", b"only once").json()["results"][0]
    wait_for_job(client, auth, r["job_id"])
    from studio.jobs import run_job
    before = client.get(f"/ingestion-jobs/{r['job_id']}", headers=auth).json()
    run_job(r["job_id"])  # a duplicate delivery (e.g. queue redelivery) must be a no-op
    assert client.get(f"/ingestion-jobs/{r['job_id']}", headers=auth).json()["attempts"] == before["attempts"]


def test_interrupted_jobs_are_requeued_on_restart(client, auth, project, monkeypatch):
    from studio.jobs import recover_interrupted
    from studio.models import DocumentVersion
    r = upload(client, auth, project["id"], "crash.txt", b"survives a restart").json()["results"][0]
    wait_for_job(client, auth, r["job_id"])
    with dbmod.SessionLocal() as db:  # simulate a crash mid-run: job says running, work not done
        j = db.get(Job, r["job_id"])
        j.status, j.attempts = "running", 1
        v = db.get(DocumentVersion, r["version_id"])
        v.status, v.extraction_key = "processing", None
        db.commit()
    assert recover_interrupted() == 1
    assert wait_for_job(client, auth, r["job_id"])["status"] == "succeeded"
    assert client.get(f"/documents/{r['document_id']}", headers=auth).json()["latest_version"]["status"] == "ready"


def test_metadata_tags_filters_and_collections(client, auth, project):
    pid = project["id"]
    col = client.post(f"/projects/{pid}/collections", headers=auth, json={"name": "Specs"}).json()
    a = ingest(client, auth, pid, "alpha.txt", b"alpha body", collection_id=col["id"], tags="spec, draft")
    ingest(client, auth, pid, "beta.md", b"beta body")
    p = client.patch(f"/documents/{a['document_id']}", headers=auth,
                     json={"tags": ["final"], "doc_metadata": {"owner": "ada"}}).json()
    assert p["tags"] == ["final"] and p["doc_metadata"] == {"owner": "ada"}
    q = lambda **kw: client.get(f"/projects/{pid}/documents", headers=auth, params=kw).json()  # noqa: E731
    assert [i["filename"] for i in q(q="alp")["items"]] == ["alpha.txt"]
    assert [i["filename"] for i in q(ext="md")["items"]] == ["beta.md"]
    assert [i["filename"] for i in q(collection_id=col["id"])["items"]] == ["alpha.txt"]
    assert [i["filename"] for i in q(tag="final")["items"]] == ["alpha.txt"]
    assert q(status="failed")["total"] == 0


def test_ingestion_stats_dashboard(client, auth, project, sample_pdf):
    pid = project["id"]
    ingest(client, auth, pid, "s.pdf", sample_pdf)
    ingest(client, auth, pid, "t.txt", b"some text here")
    s = client.get(f"/projects/{pid}/ingestion-stats", headers=auth).json()
    assert s["documents"] == 2 and s["pages"] == 4 and s["by_type"] == {"pdf": 1, "txt": 1}
    assert s["extraction_success_rate"] == 1.0 and s["queue_depth"] == 0 and s["tokens"] > 0
    assert s["storage_bytes"] == len(sample_pdf) + len(b"some text here")


def test_delete_document_removes_its_artifacts_and_jobs(client, auth, project):
    from studio.artifacts import get_store
    r = ingest(client, auth, project["id"], "gone.txt", b"delete me")
    d = client.get(f"/documents/{r['document_id']}", headers=auth).json()
    key = f"projects/{project['id']}/versions/{d['latest_version']['id']}/extraction.json"
    assert get_store().exists(key)
    assert client.delete(f"/documents/{d['id']}", headers=auth).status_code == 204
    assert not get_store().exists(key)
    assert client.get(f"/documents/{d['id']}", headers=auth).status_code == 404
    assert client.get(f"/projects/{project['id']}/ingestion-jobs", headers=auth).json()["items"] == []


def test_documents_of_another_user_are_invisible(client, auth, project):
    from conftest import register
    r = ingest(client, auth, project["id"], "private.txt", b"secret")
    other = register(client, "eve@example.com", "Eve")
    for path in (f"/documents/{r['document_id']}", f"/documents/{r['document_id']}/content",
                 f"/documents/{r['document_id']}/source"):
        assert client.get(path, headers=other).status_code == 404
    assert client.delete(f"/documents/{r['document_id']}", headers=other).status_code == 404
    assert client.get(f"/ingestion-jobs/{r['job_id']}", headers=other).status_code == 404
    assert client.get(f"/projects/{project['id']}/documents", headers=other).status_code == 404


def test_many_documents_in_one_request(client, auth, project):
    files = [("files", (f"doc{i}.txt", f"document number {i} with unique text {i * 7919}".encode(), "text/plain"))
             for i in range(40)]
    r = client.post(f"/projects/{project['id']}/documents", headers=auth, files=files).json()["results"]
    assert len(r) == 40 and all(x["status"] == "queued" for x in r)
    t0 = time.time()
    while time.time() - t0 < 60:
        s = client.get(f"/projects/{project['id']}/ingestion-stats", headers=auth).json()
        if s["by_status"].get("ready", 0) == 40 and s["queue_depth"] == 0:  # a version turns ready just before its job closes
            break
        time.sleep(0.1)
    assert s["by_status"].get("ready") == 40 and s["queue_depth"] == 0
