"""Recommendations for every pipeline stage, from what is actually in the project.

Each stage gets a pick, why, and the numbers behind it. `basis` says where the pick comes from:
  measured   your own evaluation runs support it (the evidence lists them)
  rule       a documented rule applied to measurements of your documents
  default    a safe default because there is not enough information yet
Measured evidence overrides rules. Nothing here calls a model; the same project always gets the same advice.
"""
import json
from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .artifacts import get_store
from .chunking.canon import build_canon
from .chunking.profile import choose_method, describe, profile
from .config import settings
from .embeddings import PROVIDERS
from .generation import service as gen
from .models import Document, EvalDataset, EvalQuestion, EvalRun, Index
from .registry import registry
from .retrieval.rerank_local import status as rerank_status

_profiles: dict[str, dict] = {}
MIN_N = 10  # questions needed before a measured difference is called a difference


def _doc_profile(ver) -> dict | None:
    if ver.id in _profiles:
        return _profiles[ver.id]
    if not ver.extraction_key:
        return None
    pages = json.loads(get_store().get(ver.extraction_key))
    p = profile(build_canon(pages))
    _profiles[ver.id] = p
    return p


def _stage(id_: str, title: str, method: str, label: str, basis: str, reasons: list[str], evidence: list[dict] | None = None,
           alternatives: list[dict] | None = None, params: dict | None = None, action: dict | None = None) -> dict:
    return {"id": id_, "title": title, "pick": {"method": method, "label": label, "params": params or {}}, "basis": basis,
            "reasons": reasons, "evidence": evidence or [], "alternatives": alternatives or [], "action": action}


def _ev(label: str, value) -> dict:
    return {"label": label, "value": value}


def _m(run: EvalRun, key: str) -> float | None:
    agg = ((run.metrics or {}).get("retrieval") or {}).get(key) or {}
    return agg.get("mean")


def _n(run: EvalRun) -> int:
    return int((run.metrics or {}).get("answerable") or 0)


def _ci(run: EvalRun, key: str):
    return (((run.metrics or {}).get("retrieval") or {}).get(key) or {}).get("ci95")


def _clearly_better(a: EvalRun, b: EvalRun, key: str = "ndcg") -> bool:
    """a beats b when both have enough questions and a's confidence interval sits above b's mean (or, with no interval, by 0.03)."""
    ma, mb = _m(a, key), _m(b, key)
    if ma is None or mb is None or min(_n(a), _n(b)) < MIN_N:
        return False
    ci = _ci(a, key)
    return (ci[0] > mb) if ci else (ma - mb >= 0.03)


def _label(r: EvalRun) -> str:
    c = r.config
    return f"{r.index_name} · {c.get('mode')} k={c.get('k')}" + (f" · rerank {c['rerank']}" if c.get("rerank") not in (None, "none") else "")


def recommend(db: Session, project_id: str) -> dict:
    docs = []
    for d in db.scalars(select(Document).where(Document.project_id == project_id)):
        v = d.versions[-1] if d.versions else None
        if v is not None and v.status == "ready":
            docs.append((d, v))
    indexes = db.scalars(select(Index).where(Index.project_id == project_id, Index.status == "ready")).all()
    ds_ids = [x.id for x in db.scalars(select(EvalDataset).where(EvalDataset.project_id == project_id))]
    runs = [r for r in db.scalars(select(EvalRun).where(EvalRun.dataset_id.in_(ds_ids or [""]), EvalRun.status == "done"))]
    idx_by_id = {i.id: i for i in indexes}
    runs = [r for r in runs if r.index_id in idx_by_id and _m(r, "ndcg") is not None]
    questions = db.scalar(select(func.count()).select_from(EvalQuestion).where(EvalQuestion.dataset_id.in_(ds_ids or [""]))) or 0

    total_tokens = sum(v.token_count or 0 for _, v in docs)
    total_chars = sum(v.char_count or 0 for _, v in docs)
    stages: list[dict] = []

    # ---------------------------------------------------------------- 1. parsing
    if not docs:
        stages.append(_stage("parsing", "Parsing", "pymupdf", "PyMuPDF", "default", ["No documents yet. PyMuPDF is the default for digital PDFs."],
                             action={"href": "/ingestion", "label": "Upload a document"}))
    else:
        scanned = [(d, v) for d, v in docs if v.ext == "pdf" and v.pages and (v.char_count or 0) / v.pages < 60]
        pdfs = [(d, v) for d, v in docs if v.ext == "pdf" and v.pages]
        thin_tables = [(d, v) for d, v in pdfs if v.pages >= 10 and (v.table_count or 0) / v.pages < 0.05 and (d, v) not in scanned]
        ev = [_ev(d.filename, f"{v.pages} pages, {(v.char_count or 0) // max(1, v.pages)} characters per page, {v.table_count or 0} tables, parsed with {v.parser}") for d, v in docs[:8]]
        if scanned:
            stages.append(_stage("parsing", "Parsing", "rapidocr", "RapidOCR (local OCR)", "rule",
                                 [f"{len(scanned)} PDF(s) have under 60 characters of text per page, which usually means scanned pages with no text layer: "
                                  f"{', '.join(d.filename for d, _ in scanned[:3])}.", "OCR reads the page image; PyMuPDF would return almost nothing."],
                                 ev, [{"method": "azure_document_intelligence", "why": "Stronger on tables in scans, billed per page"}],
                                 action={"href": "/ingestion", "label": "Re-upload with the RapidOCR parser"}))
        elif thin_tables:
            stages.append(_stage("parsing", "Parsing", "pymupdf", "PyMuPDF, then check tables in Parser Lab", "rule",
                                 [f"{len(thin_tables)} PDF(s) of 10+ pages have almost no tables detected ({', '.join(d.filename for d, _ in thin_tables[:3])}). "
                                  "PyMuPDF missed ruled tables that pdfplumber found on the NASA handbook, so compare them on a few pages."], ev,
                                 [{"method": "pdfplumber", "why": "Finds ruled tables PyMuPDF can miss; no printed-page numbers"}],
                                 action={"href": "/parsing", "label": "Compare parsers on 3 pages"}))
        else:
            stages.append(_stage("parsing", "Parsing", "pymupdf", "PyMuPDF", "rule",
                                 ["Documents have a text layer and tables were detected, so the fast default is enough."], ev,
                                 [{"method": "pdfplumber", "why": "Try it in Parser Lab if a table looks wrong"}]))

    # ---------------------------------------------------------------- 2. chunking
    per_doc = []
    for d, v in docs[:40]:
        p = _doc_profile(v)
        if p:
            m, why = choose_method(p)
            per_doc.append((d, v, p, m, why))
    by_method: dict[str, int] = defaultdict(int)
    for _, v, p, m, _ in per_doc:
        by_method[m] += p["chars"]
    chunk_runs = defaultdict(list)
    for r in runs:
        chunk_runs[idx_by_id[r.index_id].config["chunking"]["method"]].append(r)
    best_by_method = {m: max(rs, key=lambda r: _m(r, "ndcg") or 0) for m, rs in chunk_runs.items()}
    measured = None
    if len(best_by_method) >= 2:
        ranked = sorted(best_by_method.items(), key=lambda kv: -(_m(kv[1], "ndcg") or 0))
        if _clearly_better(ranked[0][1], ranked[1][1]):
            measured = ranked
    if measured:
        top_m, top_r = measured[0]
        spec = registry.get(top_m)
        stages.append(_stage("chunking", "Chunking", top_m, spec.name if spec else top_m, "measured",
                             [f"Your evaluation runs rank '{top_m}' first: nDCG {_m(top_r, 'ndcg'):.3f} against {_m(measured[1][1], 'ndcg'):.3f} for '{measured[1][0]}', "
                              f"and the gap is outside the confidence interval on {_n(top_r)} questions."],
                             [_ev(f"{m}", f"nDCG {(_m(r, 'ndcg') or 0):.3f} (n={_n(r)}) from {_label(r)}") for m, r in measured[:5]],
                             [{"method": m, "why": f"nDCG {(_m(r, 'ndcg') or 0):.3f}"} for m, r in measured[1:4]],
                             params=idx_by_id[top_r.index_id].config["chunking"]["params"], action={"href": "/evaluation", "label": "Open the evaluation"}))
    elif per_doc:
        top = max(by_method, key=lambda m: by_method[m])
        spec = registry.get(top)
        reasons = [f"Measured on your documents, '{top}' fits the largest share of the text ({by_method[top] / max(1, sum(by_method.values())):.0%})."]
        reasons += [f"{d.filename}: {m} because {why}." for d, _, _, m, why in per_doc[:6]]
        alts = [{"method": m, "why": f"fits {c / max(1, sum(by_method.values())):.0%} of the text"} for m, c in sorted(by_method.items(), key=lambda kv: -kv[1])[1:4]]
        alts.append({"method": "recursive", "why": "the safe general-purpose baseline to compare against"}) if top != "recursive" else None
        stages.append(_stage("chunking", "Chunking", top, spec.name if spec else top, "rule", reasons,
                             [_ev(d.filename, describe(p)) for d, _, p, _, _ in per_doc[:5]], alts,
                             action={"href": "/chunking", "label": "Compare it with recursive on a sample"}))
    else:
        stages.append(_stage("chunking", "Chunking", "recursive", "Recursive splitting", "default", ["No documents yet. Recursive splitting is the safe baseline."],
                             action={"href": "/ingestion", "label": "Upload a document"}))

    # ---------------------------------------------------------------- 3. embedding
    az_ok = PROVIDERS["azure_openai"].status()[0]
    fe_ok = PROVIDERS["fastembed"].status()[0]
    prov_runs = defaultdict(list)
    for r in runs:
        prov_runs[idx_by_id[r.index_id].provider].append(r)
    best_prov = {p: max(rs, key=lambda r: _m(r, "ndcg") or 0) for p, rs in prov_runs.items()}
    ev = [_ev("Tokens in your documents", f"{total_tokens:,}"), _ev("Embedding providers available",
          ", ".join(k for k, p in PROVIDERS.items() if p.status()[0]))]
    if len(best_prov) >= 2 and _clearly_better(*sorted(best_prov.values(), key=lambda r: -(_m(r, "ndcg") or 0))[:2]):
        top_p = max(best_prov, key=lambda p: _m(best_prov[p], "ndcg") or 0)
        stages.append(_stage("embedding", "Embedding", top_p, top_p, "measured",
                             [f"'{top_p}' scored best in your runs (nDCG {_m(best_prov[top_p], 'ndcg'):.3f})."],
                             ev + [_ev(p, f"nDCG {(_m(r, 'ndcg') or 0):.3f} (n={_n(r)})") for p, r in best_prov.items()]))
    elif az_ok:
        stages.append(_stage("embedding", "Embedding", "azure_openai", f"Azure OpenAI ({PROVIDERS['azure_openai'].model})", "rule",
                             ["It is configured and semantic. Results are cached by text, so re-indexing unchanged text is free.",
                              f"Your documents hold about {total_tokens:,} tokens, which is the billed volume for one full index."], ev,
                             [{"method": "fastembed", "why": "Free and private, but a smaller model; compare it in the evaluation grid"}]))
    elif fe_ok:
        stages.append(_stage("embedding", "Embedding", "fastembed", f"FastEmbed ({PROVIDERS['fastembed'].model})", "rule",
                             ["Azure OpenAI is not configured, so the local semantic model is the best available. It needs no key and costs nothing."], ev,
                             [{"method": "azure_openai", "why": "Larger model; add credentials in Settings to compare"}]))
    else:
        stages.append(_stage("embedding", "Embedding", "local_hash", "Offline hashing (demo only)", "default",
                             ["No semantic embedding provider is available. The hashing embedder only matches shared words and must not be used to judge quality.",
                              "Install fastembed (pip install fastembed) for free local semantic embeddings."], ev,
                             action={"href": "/settings", "label": "Open Settings"}))

    # ---------------------------------------------------------------- 4. vector store
    est_vectors = round(total_tokens / 400 * 1.1) if total_tokens else 0
    store_runs = defaultdict(list)
    for i in indexes:
        store_runs[i.store].append(i)
    if est_vectors <= 50_000:
        stages.append(_stage("vector_store", "Vector store", "qdrant", "Qdrant (embedded)", "rule",
                             [f"About {est_vectors:,} vectors is small enough for any store; Qdrant's HNSW index and payload filters are a good default.",
                              "FAISS (exact) gives the recall ceiling: use it once to check that approximate search loses nothing on your data."],
                             [_ev("Estimated vectors at 400 tokens per chunk", f"{est_vectors:,}"), _ev("Indexes by store", ", ".join(f"{s}: {len(v)}" for s, v in store_runs.items()) or "none")],
                             [{"method": "faiss", "why": "Exact search; the baseline for recall"}, {"method": "pgvector", "why": "If you want vectors beside relational data"}]))
    else:
        has_server = bool(settings.qdrant_url)
        stages.append(_stage("vector_store", "Vector store", "qdrant", "Qdrant server" if has_server else "Qdrant (run the server)", "rule",
                             [f"About {est_vectors:,} vectors is past what embedded stores handle comfortably; a server-mode store keeps search fast and concurrent.",
                              "Set QDRANT_URL to a Qdrant container (docker compose provides one)." if not has_server else "QDRANT_URL is set, so the server is already in use."],
                             [_ev("Estimated vectors at 400 tokens per chunk", f"{est_vectors:,}"), _ev("Qdrant mode", "server" if has_server else "embedded")],
                             [{"method": "pgvector", "why": "HNSW inside PostgreSQL, good with SQL filters"}, {"method": "faiss", "why": "Exact search, in memory; fine to a few hundred thousand vectors"}],
                             action={"href": "/settings", "label": "Check Qdrant in Settings"}))

    # ---------------------------------------------------------------- 5. retrieval
    if runs:
        best = max(runs, key=lambda r: _m(r, "ndcg") or 0)
        rest = [r for r in runs if r is not best]
        sure = bool(rest) and all(_clearly_better(best, r) for r in rest if (r.config.get("mode"), r.config.get("k")) != (best.config.get("mode"), best.config.get("k")))
        c = best.config
        stages.append(_stage("retrieval", "Retrieval", c.get("mode", "dense"), f"{c.get('mode')} with k = {c.get('k')}",
                             "measured" if sure else "rule",
                             [f"Best of {len(runs)} runs: {_label(best)}, nDCG {_m(best, 'ndcg'):.3f}, hit rate {(_m(best, 'hit') or 0):.2f}, MRR {(_m(best, 'mrr') or 0):.3f} on {_n(best)} questions."
                              + ("" if sure else " Other runs are within the noise of this one, so treat it as a leader, not a winner.")],
                             [_ev(_label(r), f"nDCG {(_m(r, 'ndcg') or 0):.3f}") for r in sorted(runs, key=lambda r: -(_m(r, 'ndcg') or 0))[:6]],
                             action={"href": "/evaluation", "label": "Open the evaluation"}))
    else:
        stages.append(_stage("retrieval", "Retrieval", "hybrid", "Hybrid (dense + BM25) with k = 5", "default",
                             ["Technical documents contain exact names and numbers that keyword search catches and embeddings blur; fusing both is the robust start.",
                              "Run the experiment grid (dense, BM25, hybrid) to replace this default with a measurement."],
                             action={"href": "/evaluation", "label": "Create a dataset and run the grid"}))

    # ---------------------------------------------------------------- 6. reranking
    ce_ok = rerank_status()[0]
    pairs = []
    for r in runs:
        base = next((o for o in runs if o.index_id == r.index_id and o.config.get("mode") == r.config.get("mode") and o.config.get("k") == r.config.get("k")
                     and o.config.get("rerank") == "none" and r.config.get("rerank") not in (None, "none")), None)
        if base:
            pairs.append((r, base))
    if pairs:
        gains = [((_m(r, "ndcg") or 0) - (_m(b, "ndcg") or 0), r, b) for r, b in pairs]
        g, r, b = max(gains, key=lambda x: x[0])
        helps = _clearly_better(r, b)
        stages.append(_stage("reranking", "Reranking", r.config["rerank"] if helps else "none", f"{r.config['rerank']} reranker" if helps else "No reranker", "measured",
                             [f"On the same index, mode and k, reranking with {r.config['rerank']} changed nDCG by {g:+.3f} ({(_m(b, 'ndcg') or 0):.3f} to {(_m(r, 'ndcg') or 0):.3f})."
                              + (" That is outside the noise: keep it." if helps else " That is within the noise: it is not worth the extra time or tokens yet.")],
                             [_ev(f"{x[1].config['rerank']} on {x[1].index_name}", f"{x[0]:+.3f} nDCG") for x in gains[:4]]))
    elif ce_ok:
        stages.append(_stage("reranking", "Reranking", "cross_encoder", "Cross-encoder (local, free)", "default",
                             ["A cross-encoder re-reads the top candidates with the question and is often the cheapest accuracy gain. It runs locally with no tokens.",
                              "No run compares reranked against plain retrieval yet, so this is a suggestion to test, not a measured result."],
                             action={"href": "/evaluation", "label": "Run the grid with rerank on and off"}))
    else:
        stages.append(_stage("reranking", "Reranking", "none", "No reranker", "default", ["No reranker is available. pip install fastembed enables the local cross-encoder."]))

    # ---------------------------------------------------------------- 7. generation
    gen_ok = gen.generation_status()[0]
    gate = None
    for r in sorted(runs, key=lambda r: r.created_at, reverse=True):
        sweep = ((r.metrics or {}).get("unanswerable_gate") or {}).get("sweep")
        if sweep:
            gate = (r, sweep)
            break
    if not gen_ok:
        stages.append(_stage("generation", "Answer generation", "none", "Not configured", "default",
                             ["No chat model is configured, so answers cannot be generated. Add Azure OpenAI (or another provider) in Settings."],
                             action={"href": "/settings", "label": "Open Settings"}))
    else:
        reasons = ["Grounded mode with citation checking: the model may only use the retrieved passages, and every [n] is checked against what was sent.",
                   "Judge faithfulness on a sample: a resolving citation proves the number exists, not that the passage supports the claim."]
        ev = [_ev("Default model", settings.azure_chat_deployment or "n/a")]
        params = {"strict": True}
        if gate:
            r, sweep = gate
            ok_rows = [s for s in sweep if s["unanswerable_refused"] >= 0.8]
            pick = max(ok_rows, key=lambda s: s["answerable_kept"]) if ok_rows else None
            if pick:
                params["min_score"] = pick["threshold"]
                reasons.append(f"Refuse when the best dense score is below {pick['threshold']}: on {r.index_name} that kept {pick['answerable_kept']:.0%} of answerable questions "
                               f"and refused {pick['unanswerable_refused']:.0%} of unanswerable ones.")
                ev += [_ev(f"score >= {s['threshold']}", f"keeps {s['answerable_kept']:.0%} answerable, refuses {s['unanswerable_refused']:.0%} unanswerable") for s in sweep[:6]]
        else:
            reasons.append("Add unanswerable questions to the evaluation set to choose a score threshold from data instead of guessing.")
        stages.append(_stage("generation", "Answer generation", "azure_openai_chat", "Grounded answers, strict mode", "measured" if gate else "rule",
                             reasons, ev, params=params, action={"href": "/retrieval", "label": "Ask a question"}))

    # ---------------------------------------------------------------- 8. evaluation
    if not ds_ids:
        stages.append(_stage("evaluation", "Evaluation", "dataset", "Create a question set", "default",
                             ["Nothing above can be confirmed without questions whose answers you know. Start with 20 to 30 real questions, a few of them unanswerable."],
                             action={"href": "/evaluation", "label": "Create a dataset"}))
    else:
        r = []
        if questions < 30:
            r.append(f"{questions} question(s) is too few for tight confidence intervals; aim for 30 or more.")
        if len(runs) < 4:
            r.append("Few runs so far. The experiment grid runs many configurations on the same questions in one go.")
        if not any((x.config or {}).get("with_answers") for x in runs):
            r.append("No run has generated answers yet, so faithfulness and citation checks are untested.")
        stages.append(_stage("evaluation", "Evaluation", "grid", "Run the experiment grid", "rule" if r else "measured",
                             r or ["Your evaluation covers retrieval and answers with enough questions."],
                             [_ev("Questions", questions), _ev("Completed runs", len(runs))], action={"href": "/evaluation", "label": "Open the evaluation"}))

    # ---------------------------------------------------------------- experiments to run next
    todo = []
    chunk_pick = next((s for s in stages if s["id"] == "chunking"), None)
    built = {i.config["chunking"]["method"] for i in indexes}
    if chunk_pick and chunk_pick["pick"]["method"] not in built and docs:
        todo.append(f"Build an index with '{chunk_pick['pick']['method']}' and compare it with your current one in the evaluation grid.")
    if docs and "recursive" not in built:
        todo.append("Build a plain 'recursive' index as the baseline every other method must beat.")
    if docs and not any(i.store == "faiss" for i in indexes) and est_vectors <= 200_000:
        todo.append("Build the same index on FAISS (exact) to check that approximate search loses no recall.")
    if docs and not any(i.config.get("contextual") for i in indexes) and gen_ok:
        todo.append("Try contextual embedding on one index; it costs one chat call per chunk, so use a subset first.")
    if indexes and ce_ok and not any(r.config.get("rerank") == "cross_encoder" for r in runs):
        todo.append("Run the grid with the local cross-encoder reranker on and off.")
    return {"documents": len(docs), "indexes": len(indexes), "runs": len(runs), "questions": questions, "tokens": total_tokens, "chars": total_chars,
            "stages": stages, "next_experiments": todo}


def recommend_document(db: Session, version) -> dict:
    """Parser and chunking advice for one document, with the measurements behind it."""
    p = _doc_profile(version)
    out: dict = {"profile": p}
    if p is None:
        return out | {"chunking": None}
    method, why = choose_method(p)
    spec = registry.get(method)
    out["chunking"] = {"method": method, "name": spec.name if spec else method, "why": why, "measured": describe(p),
                       "needs_layout_data": method == "layout_aware"}
    cpp = (version.char_count or 0) / version.pages if version.pages else None
    if version.ext == "pdf" and cpp is not None and cpp < 60:
        out["parser"] = {"method": "rapidocr", "why": f"only {cpp:.0f} characters per page: likely scanned"}
    return out
