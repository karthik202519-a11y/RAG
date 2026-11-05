import numpy as np
import pytest

from conftest import register
from studio.routers.explorer import pca_2d
from test_indexes import build, make_docs  # noqa: F401


@pytest.fixture()
def idx(client, auth, project):
    make_docs(client, auth, project)
    return build(client, auth, project)


def chunk_ids(client, auth, idx, n=6):
    return [c["id"] for c in client.get(f"/indexes/{idx['id']}/chunks", headers=auth, params={"limit": n}).json()["items"]]


def test_pca_recovers_the_dominant_direction():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(200, 1))
    x = np.hstack([t * 5, t * 5 + rng.normal(scale=0.1, size=(200, 1)), rng.normal(scale=0.1, size=(200, 3))])
    coords, ratio = pca_2d(x)
    assert coords.shape == (200, 2) and ratio[0] > 0.95 and ratio[0] + ratio[1] <= 1.0001
    assert abs(np.corrcoef(coords[:, 0], t[:, 0])[0, 1]) > 0.99


def test_explore_projects_a_deterministic_sample(client, auth, idx):
    a = client.get(f"/indexes/{idx['id']}/explore", headers=auth, params={"sample": 5, "seed": 3}).json()
    b = client.get(f"/indexes/{idx['id']}/explore", headers=auth, params={"sample": 5, "seed": 3}).json()
    assert a == b and a["sampled"] == 5 and a["dimension"] == 256 and len(a["points"]) == 5
    everything = client.get(f"/indexes/{idx['id']}/explore", headers=auth, params={"sample": 500}).json()
    assert everything["sampled"] == everything["total_chunks"]  # asking for more than exists returns all of it
    assert 0 < sum(a["explained_variance"]) <= 1.0 and "distorts" in a["warning"]
    assert {"x", "y", "filename", "chunk_id", "preview"} <= set(a["points"][0])
    one = client.get(f"/indexes/{idx['id']}/explore", headers=auth, params={"sample": 50, "document_id": "nope"})
    assert one.status_code == 422


def test_similarity_measures_agree_for_unit_vectors(client, auth, idx):
    ids = chunk_ids(client, auth, idx, 6)
    r = client.post(f"/indexes/{idx['id']}/similarity", headers=auth, json={"query": "Saturn V Apollo rocket", "chunk_ids": ids}).json()
    assert r["dimension"] == 256 and abs(r["query_norm"] - 1) < 1e-4
    for row in r["rows"]:
        assert row["cosine"] == pytest.approx(row["dot"], abs=1e-4)
        assert row["euclidean"] ** 2 == pytest.approx(2 - 2 * row["cosine"], abs=1e-3)
        assert row["rank_cosine"] == row["rank_dot"] == row["rank_euclidean"] or abs(row["cosine"]) < 1e-9
    assert sorted(x["rank_cosine"] for x in r["rows"]) == list(range(1, 7))


def test_similarity_matrix_is_symmetric_with_unit_diagonal(client, auth, idx):
    ids = chunk_ids(client, auth, idx, 5)
    m = client.post(f"/indexes/{idx['id']}/similarity-matrix", headers=auth, json={"chunk_ids": ids}).json()["matrix"]
    a = np.array(m)
    assert a.shape == (5, 5) and np.allclose(a, a.T, atol=1e-4) and np.allclose(np.diag(a), 1.0, atol=1e-3)


def test_explorer_rejects_foreign_chunks_and_other_users(client, auth, project, idx):
    ids = chunk_ids(client, auth, idx, 2)
    bad = client.post(f"/indexes/{idx['id']}/similarity", headers=auth, json={"query": "x", "chunk_ids": ["0" * 32]})
    assert bad.status_code == 404
    eve = register(client, "eve@example.com", "Eve")
    assert client.get(f"/indexes/{idx['id']}/explore", headers=eve).status_code == 404
    assert client.post(f"/indexes/{idx['id']}/similarity", headers=eve, json={"query": "x", "chunk_ids": ids}).status_code == 404
    assert client.post(f"/indexes/{idx['id']}/similarity-matrix", headers=eve, json={"chunk_ids": ids}).status_code == 404
    assert client.post(f"/indexes/{idx['id']}/similarity", headers=auth, json={"query": "x", "chunk_ids": []}).status_code == 422
