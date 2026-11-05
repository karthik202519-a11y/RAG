"""Local cross-encoder reranking (fastembed ONNX). Unlike the bi-encoder used for search, a cross-encoder reads the question and a
passage together, which is slower but usually more accurate. Free, offline after the model downloads, no chat-model tokens."""
import math
import threading

from ..config import settings

_lock = threading.Lock()
_models: dict = {}


def status() -> tuple[bool, str]:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False, "pip install fastembed"
    return True, f"local model {settings.fastembed_rerank_model} (downloaded on first use)"


def _model():
    with _lock:
        m = _models.get(settings.fastembed_rerank_model)
        if m is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            m = _models[settings.fastembed_rerank_model] = TextCrossEncoder(
                model_name=settings.fastembed_rerank_model, cache_dir=str(settings.model_cache_dir))
        return m


def rerank_cross_encoder(question: str, hits: list[dict]) -> tuple[list[dict], str | None]:
    """Reorder `hits` by cross-encoder score. Returns (hits, warning); on failure the input order is kept with a warning."""
    if not hits:
        return [], None
    ok, why = status()
    if not ok:
        return hits, f"Cross-encoder reranking is unavailable: {why}"
    try:
        scores = [float(s) for s in _model().rerank(question, [h["text"][:2000] for h in hits])]
    except Exception as e:
        return hits, f"Cross-encoder failed ({type(e).__name__}); original order kept."
    if len(scores) != len(hits) or not all(math.isfinite(s) for s in scores):
        return hits, "Cross-encoder returned invalid scores; original order kept."
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    out = []
    for new_rank, i in enumerate(order, 1):
        h = dict(hits[i])
        h["rerank"] = {"score": round(scores[i], 3), "before_rank": i + 1, "after_rank": new_rank, "moved": (i + 1) - new_rank, "method": "cross_encoder"}
        out.append(h)
    return out, None
