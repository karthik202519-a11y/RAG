from sqlalchemy import select

from conftest import register
from studio import db as dbmod
from studio.models import User, WorkspaceMember


def test_register_login_me(client):
    h = register(client)
    assert client.get("/auth/me", headers=h).json()["email"] == "a@example.com"
    r = client.post("/auth/login", json={"email": "A@Example.com", "password": "correct horse battery"})
    assert r.status_code == 200 and r.json()["token"]


def test_duplicate_email_and_weak_password(client):
    register(client)
    r = client.post("/auth/register", json={"email": "a@example.com", "password": "another long pass", "display_name": "x"})
    assert r.status_code == 409
    r = client.post("/auth/register", json={"email": "b@example.com", "password": "short", "display_name": "x"})
    assert r.status_code == 422


def test_wrong_password_and_lockout(client):
    register(client)
    for _ in range(5):
        assert client.post("/auth/login", json={"email": "a@example.com", "password": "wrong password"}).status_code == 401
    assert client.post("/auth/login", json={"email": "a@example.com", "password": "correct horse battery"}).status_code == 429


def test_passwords_are_not_stored_in_plaintext(client):
    register(client)
    with dbmod.SessionLocal() as db:
        u = db.scalar(select(User))
        assert "correct horse" not in u.password_hash and u.password_hash.startswith("scrypt$")


def test_protected_routes_need_a_valid_token(client):
    assert client.get("/projects").status_code == 401
    assert client.get("/projects", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    assert client.get("/method-catalog").status_code == 200  # the catalog is public reference data


def test_projects_are_isolated_between_users(client):
    a = register(client, "a@example.com", "Ada")
    b = register(client, "b@example.com", "Bob")
    ws_a = client.get("/workspaces", headers=a).json()[0]
    p = client.post("/projects", headers=a, json={"workspace_id": ws_a["id"], "name": "Secret"}).json()
    assert client.get("/projects", headers=b).json() == []
    # both "exists but forbidden" and "does not exist" give 404, so ids cannot be probed
    assert client.get(f"/projects/{p['id']}", headers=b).status_code == 404
    assert client.get("/projects/doesnotexist", headers=b).status_code == 404
    assert client.delete(f"/projects/{p['id']}", headers=b).status_code == 404
    assert client.post(f"/projects/{p['id']}/collections", headers=b, json={"name": "x"}).status_code == 404
    ws_b = client.get("/workspaces", headers=b).json()[0]
    assert client.post("/projects", headers=b, json={"workspace_id": ws_a["id"], "name": "x"}).status_code == 404
    assert client.post("/projects", headers=b, json={"workspace_id": ws_b["id"], "name": "mine"}).status_code == 201


def test_viewer_role_is_read_only(client):
    a = register(client, "a@example.com", "Ada")
    b = register(client, "b@example.com", "Bob")
    ws_a = client.get("/workspaces", headers=a).json()[0]
    p = client.post("/projects", headers=a, json={"workspace_id": ws_a["id"], "name": "Shared"}).json()
    with dbmod.SessionLocal() as db:
        bob = db.scalar(select(User).where(User.email == "b@example.com"))
        db.add(WorkspaceMember(workspace_id=ws_a["id"], user_id=bob.id, role="viewer"))
        db.commit()
    assert client.get(f"/projects/{p['id']}", headers=b).status_code == 200
    r = client.post(f"/projects/{p['id']}/collections", headers=b, json={"name": "x"})
    assert r.status_code == 403
    r = client.post(f"/projects/{p['id']}/documents", headers=b, files=[("files", ("a.txt", b"hello", "text/plain"))])
    assert r.status_code == 403
    assert client.delete(f"/projects/{p['id']}", headers=b).status_code == 403


def test_collections(client, auth, project):
    r = client.post(f"/projects/{project['id']}/collections", headers=auth, json={"name": "Handbooks"})
    assert r.status_code == 201
    assert [c["name"] for c in client.get(f"/projects/{project['id']}/collections", headers=auth).json()] == ["Handbooks"]
