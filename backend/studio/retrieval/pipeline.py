"""Multi-stage retrieval: query transform -> retrieve -> fuse -> MMR -> rerank.

Each stage is reported separately in `stages`, with its own timing and any LLM tokens, so a query
transformation is never confused with the search itself and every reordering can be attributed.
LLM stages use the configured Azure chat deployment and need it; the others do not.
"""
import json
import re
import time
import uuid

import numpy as np
from sqlalchemy.orm import Session

from .. import generation as gen_pkg
from ..embeddings import embed_texts
from ..generation import service as gen
from ..models import Index
from ..vectorstores import store_for
from . import service
from .expand import expand_results

TRANSFORMS = ("none", "rewrite", "multi_query", "hyde", "decompose")


class PipelineError(RuntimeError):
    pass


def _llm(model: str | None, system: str, user: str, max_tokens: int = 400) -> tuple[str, int]:
    ok, why = gen.status_for(model or gen.settings.azure_chat_deployment)
    if not ok:
        raise gen_pkg.GenerationNotConfigured(why)
    res = gen.chat_complete(model or gen.settings.azure_chat_deployment,
                            [{"role": "system", "content": system}, {"role": "user", "content": user}], 0.0, max_tokens)
    return res.text.strip(), res.prompt_tokens + res.completion_tokens


def _lines(text: str, limit: int) -> list[str]:
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip().strip('"')
        if ln and ln not in out:
            out.append(ln)
    return out[:limit]


def transform_query(kind: str, question: str, n: int, model: str | None, mode: str = "dense") -> tuple[list[dict], int]:
    """Return [{"text", "role", "search", "mode"?}]. `search` says whether the query is actually run; `mode` overrides the retriever.

    A rewrite replaces the original. HyDE embeds the hypothetical passage instead of the question (dense), while keyword
    search keeps the original, because a made-up passage has no business in keyword matching.
    """
    orig = {"text": question, "role": "original", "search": True}
    if kind == "none":
        return [orig], 0
    if kind == "rewrite":
        t, tok = _llm(model, "Rewrite the user's question as one standalone, specific search query for a technical document. "
                             "Keep identifiers and acronyms exactly. Return only the query.", question, 120)
        return ([{**orig, "search": not t}] + ([{"text": t, "role": "rewrite", "search": True}] if t else [])), tok
    if kind == "multi_query":
        t, tok = _llm(model, f"Write {n} different search queries that would each retrieve passages answering the question. "
                             "One per line, no numbering, no commentary.", question, 200)
        return [orig] + [{"text": q, "role": "variant", "search": True} for q in _lines(t, n)], tok
    if kind == "hyde":
        t, tok = _llm(model, "Write a short passage (3-4 sentences) that would answer the question, in the style of a technical "
                             "handbook. State it plainly; do not hedge.", question, 220)
        if not t or mode == "bm25":
            return [orig] + ([{"text": t, "role": "hypothetical_passage", "search": False}] if t else []), tok
        if mode == "dense":
            return [{**orig, "search": False}, {"text": t, "role": "hypothetical_passage", "search": True, "mode": "dense"}], tok
        return [{**orig, "mode": "bm25"}, {"text": t, "role": "hypothetical_passage", "search": True, "mode": "dense"}], tok
    if kind == "decompose":
        t, tok = _llm(model, f"If the question needs several separate facts, split it into at most {n} standalone sub-questions, one "
                             "per line. If it is already simple, return it unchanged on one line.", question, 200)
        subs = _lines(t, n)
        return [orig] + [{"text": q, "role": "sub_question", "search": True} for q in subs if q != question], tok
    raise PipelineError(f"Unknown query transform '{kind}'")


def rerank_llm(question: str, hits: list[dict], model: str | None) -> tuple[list[dict], int, str | None]:
    """Ask the chat model to score each passage 0-10 and reorder. Falls back to the input order, with a warning."""
    numbered = "\n\n".join(f"[{i + 1}] {h['text'][:700]}" for i, h in enumerate(hits))
    text, tok = _llm(model, "You rate how well each numbered passage helps answer the question. "
                            f"Output ONLY a JSON array of {len(hits)} integers from 0 (useless) to 10 (directly answers), in passage order.",
                     f"Question: {question}\n\nPassages:\n\n{numbered}", 200)
    m = re.search(r"\[[\d\s,.]+\]", text)
    try:
        scores = [float(x) for x in json.loads(m.group(0))] if m else []
    except (ValueError, json.JSONDecodeError):
        scores = []
    if len(scores) != len(hits):
        return hits, tok, f"The reranker returned {len(scores)} scores for {len(hits)} passages; original order kept."
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    out = []
    for new_rank, i in enumerate(order, 1):
        h = dict(hits[i])
        h["rerank"] = {"score": scores[i], "before_rank": i + 1, "after_rank": new_rank, "moved": (i + 1) - new_rank}
        out.append(h)
    return out, tok, None


def _fetch_vectors(index: Index, chunk_ids: list[str]) -> dict[str, np.ndarray]:
    return store_for(index.store).get_vectors(index.collection, chunk_ids)


def mmr(index: Index, hits: list[dict], qvec: np.ndarray, top_k: int, lam: float) -> list[dict]:
    """Maximal marginal relevance: pick results that are relevant to the query but unlike those already picked."""
    vecs = _fetch_vectors(index, [h["chunk_id"] for h in hits])
    cand = [h for h in hits if h["chunk_id"] in vecs]
    if len(cand) <= 1:
        return hits[:top_k]
    rel = {h["chunk_id"]: float(vecs[h["chunk_id"]] @ qvec) for h in cand}
    chosen: list[dict] = []
    left = list(cand)
    while left and len(chosen) < top_k:
        def score(h):
            red = max((float(vecs[h["chunk_id"]] @ vecs[c["chunk_id"]]) for c in chosen), default=0.0)
            return lam * rel[h["chunk_id"]] - (1 - lam) * red
        best = max(left, key=score)
        picked = dict(best)
        picked["mmr"] = {"relevance": round(rel[best["chunk_id"]], 4),
                         "max_similarity_to_selected": round(max((float(vecs[best["chunk_id"]] @ vecs[c["chunk_id"]]) for c in chosen), default=0.0), 4)}
        chosen.append(picked)
        left.remove(best)
    return chosen


def run(db: Session, index: Index, question: str, *, mode: str = "dense", top_k: int = 5, transform: str = "none",
        n_queries: int = 3, mmr_on: bool = False, mmr_lambda: float = 0.6, rerank: str = "none", rerank_candidates: int = 20,
        llm_model: str | None = None, document_ids=None, types=None, score_threshold=None, rrf_k: int = 60,
        expand: str = "none", neighbors: int = 1) -> dict:
    stages: list[dict] = []
    llm_tokens = 0
    cand = max(top_k * 4, rerank_candidates if rerank != "none" else 0, 20)

    # 1. query transformation (LLM, optional)
    t0 = time.perf_counter()
    queries, tok = transform_query(transform, question, n_queries, llm_model, mode)
    llm_tokens += tok
    stages.append({"stage": "query_transform", "method": transform, "queries": queries, "llm_tokens": tok,
                   "ms": round((time.perf_counter() - t0) * 1000, 1)})

    # 2. retrieval, once per query; HyDE embeds the hypothetical passage but keywords use the original question
    t0 = time.perf_counter()
    per_query = []
    qvec = None
    for q in queries:
        if not q.get("search", True):
            continue
        qmode = q.get("mode", mode)
        text = q["text"]
        r = service.search(db, index, text, qmode, cand, cand, None, document_ids, types, rrf_k)
        per_query.append({"query": text, "role": q["role"], "mode": qmode, "n": len(r["results"]), "results": r["results"],
                          "info": r["info"], "timings_ms": r["timings_ms"]})
    stages.append({"stage": "retrieve", "mode": mode, "candidates_per_query": cand,
                   "per_query": [{k: v for k, v in pq.items() if k not in ("results", "info")} for pq in per_query],
                   "ms": round((time.perf_counter() - t0) * 1000, 1)})

    # 3. fuse the per-query rankings (reciprocal rank fusion); with one query this is the identity
    t0 = time.perf_counter()
    if len(per_query) == 1:
        fused = [dict(h) for h in per_query[0]["results"]]
        for h in fused:
            h["queries_found_by"] = 1
    else:
        acc: dict[str, dict] = {}
        for pq in per_query:
            for h in pq["results"]:
                e = acc.setdefault(h["chunk_id"], {"hit": dict(h), "score": 0.0, "n": 0})
                e["score"] += 1.0 / (rrf_k + h["rank"])
                e["n"] += 1
        fused = []
        for e in sorted(acc.values(), key=lambda e: -e["score"]):
            h = e["hit"]
            h["score"], h["queries_found_by"] = round(e["score"], 6), e["n"]
            fused.append(h)
        for i, h in enumerate(fused, 1):
            h["rank"] = i
    stages.append({"stage": "fuse", "n_queries": len(per_query), "unique_chunks": len(fused), "ms": round((time.perf_counter() - t0) * 1000, 1)})
    ret_info = per_query[0]["info"] if per_query else {}

    # 4. optional diversification
    if mmr_on and fused:
        t0 = time.perf_counter()
        emb = embed_texts(db, index.provider, [question])
        qvec = emb.vectors[0]
        before = [h["chunk_id"] for h in fused[:top_k]]
        fused = mmr(index, fused, qvec, max(top_k, rerank_candidates if rerank != "none" else top_k), mmr_lambda)
        stages.append({"stage": "mmr", "lambda": mmr_lambda, "changed_top_k": before != [h["chunk_id"] for h in fused[:top_k]],
                       "ms": round((time.perf_counter() - t0) * 1000, 1)})

    # 5. optional reranking of the candidate pool
    warnings: list[str] = []
    if rerank in ("llm", "cross_encoder") and fused:
        t0 = time.perf_counter()
        pool = fused[:max(top_k, rerank_candidates)]
        if rerank == "llm":
            reranked, tok, warn = rerank_llm(question, pool, llm_model)
        else:
            from .rerank_local import rerank_cross_encoder
            (reranked, warn), tok = rerank_cross_encoder(question, pool), 0
        llm_tokens += tok
        if warn:
            warnings.append(warn)
        fused = reranked + fused[len(pool):]
        stages.append({"stage": "rerank", "method": rerank, "pool": len(pool), "llm_tokens": tok, "warning": warn,
                       "moves": [{"chunk_id": h["chunk_id"], **h["rerank"]} for h in reranked[:top_k] if "rerank" in h],
                       "ms": round((time.perf_counter() - t0) * 1000, 1)})

    final = fused
    if score_threshold is not None:
        final = [h for h in final if h["contributions"].get("dense", {}).get("score", h["score"]) >= score_threshold]
    if expand != "none":
        t0 = time.perf_counter()
        before = len(final)
        final = expand_results(db, final, expand, neighbors, limit=top_k)
        stages.append({"stage": "expand", "method": expand, "hits_in": before, "hits_out": len(final),
                       "ms": round((time.perf_counter() - t0) * 1000, 1)})
    final = final[:top_k]
    for i, h in enumerate(final, 1):
        h["final_rank"] = i
    return {"query": question, "mode": mode, "top_k": top_k, "results": final, "stages": stages, "llm_tokens": llm_tokens,
            "warnings": warnings, "info": ret_info,
            "timings_ms": {s["stage"]: s["ms"] for s in stages}}
