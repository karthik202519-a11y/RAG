from studio.config import settings


def test_overview_never_returns_secret_values(client, auth, monkeypatch):
    monkeypatch.setattr(settings, "azure_api_key", "super-secret-key-123")
    monkeypatch.setattr(settings, "azure_endpoint", "https://example.invalid")
    r = client.get("/settings/overview", headers=auth)
    assert r.status_code == 200
    assert "super-secret-key-123" not in r.text and "example.invalid" not in r.text
    states = {a["env"]: a["state"] for a in r.json()["azure"]}
    assert states["AZURE_OPENAI_API_KEY"] == "set"
    assert client.get("/settings/overview").status_code in (401, 403)


def test_local_connection_tests(client, auth):
    for target in ("database", "storage", "qdrant", "faiss"):
        r = client.post("/settings/test", headers=auth, json={"target": target}).json()
        assert r["ok"], r
    bad = client.post("/settings/test", headers=auth, json={"target": "chat"}).json()
    assert bad["ok"] is False and bad["detail"]  # not configured in tests: reported, not a 500
