import os
import sys
import tempfile
import time
from pathlib import Path

# Configure BEFORE importing studio: settings are read at import time.
_TMP = Path(tempfile.mkdtemp(prefix="studio-tests-"))
os.environ["STUDIO_DATA_DIR"] = str(_TMP)
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'boot.db').as_posix()}"
os.environ["STUDIO_JOB_BACKEND"] = "thread"
os.environ["STUDIO_SKIP_WARMUP"] = "1"
os.environ["STUDIO_SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ.pop("REDIS_URL", None)
os.environ.pop("S3_ENDPOINT", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from studio import artifacts, db as dbmod, jobs  # noqa: E402
from studio.config import settings  # noqa: E402
from studio.jobs import runner  # noqa: E402


def ensure_test_database(url: str) -> None:
    """Guard the destructive schema-reset fixture before opening its target."""
    if url.startswith("postgresql"):
        from sqlalchemy.engine import make_url
        database = (make_url(url).database or "").lower()
        if not any(marker in database for marker in ("test", "matrix")):
            raise RuntimeError("STUDIO_TEST_DATABASE_URL must name a dedicated test or matrix database; fixtures reset its public schema.")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A fresh database, artifact store and job pool for every test."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    # STUDIO_TEST_DATABASE_URL runs the suite against a real server (e.g. the Postgres in docker-compose).
    url = os.environ.get("STUDIO_TEST_DATABASE_URL") or f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    ensure_test_database(url)
    monkeypatch.setattr(settings, "database_url", url)
    dbmod.reconfigure(url)
    if url.startswith("postgresql"):
        from sqlalchemy import text
        with dbmod.engine.begin() as conn:  # start every test from an empty schema
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
    artifacts.reset_store()
    runner._pool = None
    monkeypatch.setattr(runner, "BACKOFF_BASE_SECONDS", 0.05)  # fast retries
    # each test gets its own embedded Qdrant folder; tests never talk to Azure or a Qdrant server
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path / "qdrant"))
    for name in ("azure_endpoint", "azure_api_key", "azure_embedding_deployment", "azure_vision_deployment"):
        monkeypatch.setattr(settings, name, "")
    from studio.vectorstores import qdrant_store
    qdrant_store.reset_client()
    from studio.retrieval import service as _retr
    _retr._cache.clear()
    from studio.main import app, migrate
    migrate()
    with TestClient(app) as c:
        yield c
    jobs.shutdown()
    runner._pool = None
    qdrant_store.reset_client()


def register(client, email="a@example.com", name="Ada", password="correct horse battery"):
    r = client.post("/auth/register", json={"email": email, "password": password, "display_name": name})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.fixture()
def auth(client):
    return register(client)


@pytest.fixture()
def project(client, auth):
    ws = client.get("/workspaces", headers=auth).json()[0]
    r = client.post("/projects", headers=auth, json={"workspace_id": ws["id"], "name": "Test project"})
    assert r.status_code == 201, r.text
    return r.json()


def wait_for_job(client, headers, job_id, timeout=60.0, until=("succeeded", "dead", "cancelled")):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/ingestion-jobs/{job_id}", headers=headers).json()
        if j["status"] in until:
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still {j['status']} after {timeout}s")


def upload(client, headers, project_id, name, data: bytes, **form):
    return client.post(f"/projects/{project_id}/documents", headers=headers,
                       files=[("files", (name, data, "application/octet-stream"))], data=form)


def make_pdf(pages: list[list[str]]) -> bytes:
    doc = pymupdf.open()
    for paras in pages:
        page = doc.new_page()
        y = 80
        for para in paras:
            page.insert_textbox(pymupdf.Rect(72, y, 540, y + 90), para, fontsize=11)
            y += 100
    data = doc.tobytes()
    doc.close()
    return data


SAMPLE_PAGES = [
    ["Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen inside chloroplasts. "
     "Chlorophyll absorbs mostly red and blue light and reflects green light.",
     "The Calvin cycle uses ATP and NADPH to fix carbon dioxide into sugars. It does not need light directly."],
    ["The Apollo program landed twelve astronauts on the Moon between 1969 and 1972. "
     "The Saturn V rocket carried the Apollo spacecraft toward lunar orbit.",
     "Apollo 11 landed first, on 20 July 1969, in the Sea of Tranquility."],
    ["A relational database stores rows in tables and uses SQL to query them. "
     "An index speeds up lookups at the cost of extra storage and slower writes.",
     "Transactions group several operations so that either all of them take effect or none do."],
]


@pytest.fixture()
def sample_pdf() -> bytes:
    return make_pdf(SAMPLE_PAGES)


@pytest.fixture()
def future_method(monkeypatch):
    """A chunking method that is described but not runnable, to test the honesty path now that every real method runs."""
    from studio.registry import MethodSpec, registry
    spec = MethodSpec(id="future_method", category="chunking", stage="Chunking", name="Future method", summary="s", how_it_works="h",
                      advantages=[], disadvantages=[], status="unavailable", unavailable_reason="Not built in this test.")
    monkeypatch.setitem(registry._methods, "future_method", spec)
    return spec.id
