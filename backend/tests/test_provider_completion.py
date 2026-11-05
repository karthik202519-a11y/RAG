"""Regression checks for provider routing, asymmetric embedding caches and rerank failures."""
import numpy as np
import pytest

from studio.config import settings
from studio.embeddings import providers, providers_extra, service
from studio.retrieval import rerank_local


@pytest.mark.parametrize("provider,field,response,query_type,document_type", [
    ("cohere", "cohere_api_key", {"embeddings": {"float": [[3, 4]]}}, "search_query", "search_document"),
    ("voyage", "voyage_api_key", {"data": [{"index": 0, "embedding": [3, 4]}]}, "query", "document"),
])
def test_embedding_request_roles_and_cache_isolation(client, monkeypatch, provider, field, response, query_type, document_type):
    from studio import db as dbmod
    monkeypatch.setattr(settings, field, "test-key")
    calls = []

    def post(url, headers, body, name, *args):
        calls.append(body)
        return response

    monkeypatch.setattr(providers_extra, "_post_json", post)
    with dbmod.SessionLocal() as db:
        document = service.embed_texts(db, provider, ["same text"])
        query = service.embed_texts(db, provider, ["same text"], input_type="query")
        again = service.embed_texts(db, provider, ["same text"], input_type="query")
    assert [c["input_type"] for c in calls] == [document_type, query_type]
    assert document.cache_hits == query.cache_hits == 0
    assert again.cache_hits == 1
    assert np.allclose(query.vectors, [[0.6, 0.8]])


def test_symmetric_provider_reuses_cache_between_query_and_document(client):
    from studio import db as dbmod
    with dbmod.SessionLocal() as db:
        service.embed_texts(db, "local_hash", ["rocket"])
        assert service.embed_texts(db, "local_hash", ["rocket"], input_type="query").cache_hits == 1
    with pytest.raises(ValueError, match="input_type"):
        service.embed_texts(None, "local_hash", ["x"], input_type="invalid")


def test_openai_embedding_response_order_and_normalization(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(providers_extra, "_post_json", lambda *a: {
        "data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]}],
        "usage": {"total_tokens": 9},
    })
    result = providers.get_provider("openai").embed(["a", "b"])
    assert result.tokens == 9
    assert np.allclose(result.vectors, [[1, 0], [0, 1]])


@pytest.mark.parametrize("scores", [[0.1], [float("nan"), 0.9], [float("inf"), 0.9]])
def test_reranker_invalid_results_keep_original_order(monkeypatch, scores):
    monkeypatch.setattr(rerank_local, "status", lambda: (True, "ok"))
    class Model:
        def rerank(self, question, passages):
            return scores
    monkeypatch.setattr(rerank_local, "_model", Model)
    hits = [{"text": "first"}, {"text": "second"}]
    result, warning = rerank_local.rerank_cross_encoder("question", hits)
    assert result == hits and "invalid scores" in warning


def test_reranker_empty_input_does_not_load_model(monkeypatch):
    monkeypatch.setattr(rerank_local, "status", lambda: pytest.fail("must not load"))
    assert rerank_local.rerank_cross_encoder("question", []) == ([], None)


def test_reranker_scores_reorder_and_preserve_input(monkeypatch):
    monkeypatch.setattr(rerank_local, "status", lambda: (True, "ok"))
    class Model:
        def rerank(self, question, passages):
            return [0.1, 0.9]
    monkeypatch.setattr(rerank_local, "_model", Model)
    hits = [{"text": "irrelevant"}, {"text": "answer"}]
    result, warning = rerank_local.rerank_cross_encoder("question", hits)
    assert warning is None and result[0]["text"] == "answer"
    assert result[0]["rerank"]["before_rank"] == 2
    assert "rerank" not in hits[0]


def test_ollama_embedding_request_and_response(monkeypatch):
    p = providers.get_provider("ollama")
    monkeypatch.setattr(p, "status", lambda: (True, "ok"))
    calls = []
    def post(url, headers, body, name, *args):
        calls.append((url, body))
        return {"embeddings": [[0, 2], [3, 0]]}
    monkeypatch.setattr(providers_extra, "_post_json", post)
    result = p.embed(["first", "second"])
    assert calls[0][0].endswith("/api/embed")
    assert calls[0][1]["input"] == ["first", "second"]
    assert np.allclose(result.vectors, [[0, 1], [1, 0]])


@pytest.mark.parametrize("provider,field", [("openai", "openai_api_key"), ("cohere", "cohere_api_key"), ("voyage", "voyage_api_key")])
def test_missing_embedding_credentials_fail_clearly(monkeypatch, provider, field):
    monkeypatch.setattr(settings, field, "")
    with pytest.raises(providers.NotConfigured, match="not configured"):
        providers.get_provider(provider).embed(["text"])


def test_reranker_model_failure_keeps_original_results(monkeypatch):
    monkeypatch.setattr(rerank_local, "status", lambda: (True, "ok"))
    def fail():
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(rerank_local, "_model", fail)
    hits = [{"text": "original"}]
    result, warning = rerank_local.rerank_cross_encoder("q", hits)
    assert result == hits and "original order kept" in warning


@pytest.mark.parametrize("metric", ["cosine", "dot", "euclid"])
def test_lancedb_metric_survives_reopening(client, monkeypatch, metric):
    import uuid
    from studio.vectorstores import lancedb_store as store
    monkeypatch.setattr(store, "_db", None)
    monkeypatch.setattr(store, "_meta", {})
    name = "restart_" + uuid.uuid4().hex[:8]
    store.create_collection(name, 2, metric)
    try:
        store.upsert(name, ["one", "two"], np.array([[1, 0], [0, 1]], dtype=np.float32), [{}, {}])
        before = store.search(name, np.array([1, 0], dtype=np.float32), 2)
        store._meta.pop(name)
        store._db = None
        after = store.search(name, np.array([1, 0], dtype=np.float32), 2)
        assert store._info(name)["metric"] == metric
        assert [(h.id, h.score) for h in before] == [(h.id, h.score) for h in after]
    finally:
        store.delete_collection(name)


def test_index_model_drift_rejected_even_when_dimensions_match():
    with pytest.raises(providers.ProviderError, match="differs from this index"):
        service.embed_texts(None, "local_hash", ["text"], expected_model="old-hash-same-dimension")
