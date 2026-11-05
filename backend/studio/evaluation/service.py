"""Evaluation datasets: importing labelled questions, running them against an index, comparing runs."""
import random
import statistics
import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import generation, retrieval
from ..models import Document, EvalQuestion, EvalRun, Index
from ..retrieval import pipeline
from . import metrics as M

SPLITS = ("train", "validation", "test")


class ImportError_(ValueError):
    pass


# --------------------------------------------------------------------------- import
def _native(row: dict) -> dict:
    ev = [{"document": e.get("document") or "", "pages": [int(p) for p in e.get("pages") or []]} for e in row.get("evidence") or []]
    return {"question": row["question"], "qtype": row.get("type") or row.get("qtype") or "", "evidence": ev,
            "keywords": row.get("keywords") or [], "answerable": bool(row.get("answerable", bool(ev))),
            "reference_answer": row.get("reference_answer"), "split": row.get("split") or "test"}


def _proto_basic(row: dict, doc: str) -> dict:  # eval/questions.json: {question, type, pdf_pages}
    return {"question": row["question"], "qtype": row.get("type", ""),
            "evidence": [{"document": doc, "pages": list(row.get("pdf_pages") or [])}], "keywords": [], "answerable": True,
            "reference_answer": None, "split": "test"}


def _proto_hard(row: dict) -> dict:  # eval/questions_hard.json: {cat, doc, question, page_groups, keywords}
    groups = row.get("page_groups") or []
    return {"question": row["question"], "qtype": row.get("cat", ""),
            "evidence": [{"document": row.get("doc", ""), "pages": list(g)} for g in groups],
            "keywords": row.get("keywords") or [], "answerable": row.get("cat") != "unanswerable" and bool(groups),
            "reference_answer": None, "split": "test"}


def parse_rows(rows: list, fmt: str = "auto", default_document: str = "") -> tuple[list[dict], str]:
    if not isinstance(rows, list) or not rows:
        raise ImportError_("Expected a non-empty JSON array of questions.")
    if not all(isinstance(r, dict) and r.get("question") for r in rows):
        raise ImportError_("Every item needs a non-empty 'question'.")
    if fmt == "auto":
        r0 = rows[0]
        fmt = "native" if "evidence" in r0 else "prototype_hard" if "page_groups" in r0 else "prototype_basic" if "pdf_pages" in r0 else "native"
    if fmt == "native":
        return [_native(r) for r in rows], fmt
    if fmt == "prototype_basic":
        return [_proto_basic(r, default_document) for r in rows], fmt
    if fmt == "prototype_hard":
        return [_proto_hard(r) for r in rows], fmt
    raise ImportError_(f"Unknown format '{fmt}'")


def import_questions(db: Session, dataset_id: str, project_id: str, rows: list, fmt: str, default_document: str,
                     skip_missing_documents: bool) -> dict:
    parsed, fmt = parse_rows(rows, fmt, default_document)
    names = [d.filename.lower() for d in db.scalars(select(Document).where(Document.project_id == project_id))]
    start = len(db.scalars(select(EvalQuestion.id).where(EvalQuestion.dataset_id == dataset_id)).all())
    added, skipped = 0, []
    for q in parsed:
        if q["split"] not in SPLITS:
            raise ImportError_(f"split must be one of {SPLITS}")
        docs = [e["document"] for e in q["evidence"] if e["document"]]
        if skip_missing_documents and docs and not all(any(d.lower() in n for n in names) for d in docs):
            skipped.append({"question": q["question"][:80], "reason": f"document not in this project: {', '.join(docs)}"})
            continue
        db.add(EvalQuestion(dataset_id=dataset_id, position=start + added, **q))
        added += 1
    db.commit()
    return {"format": fmt, "added": added, "skipped": skipped}


def assign_splits(db: Session, dataset_id: str, train: float, validation: float, seed: int) -> dict[str, int]:
    qs = db.scalars(select(EvalQuestion).where(EvalQuestion.dataset_id == dataset_id).order_by(EvalQuestion.position)).all()
    order = list(range(len(qs)))
    random.Random(seed).shuffle(order)
    n_train, n_val = round(len(qs) * train), round(len(qs) * validation)
    counts = {s: 0 for s in SPLITS}
    for rank, i in enumerate(order):
        split = "train" if rank < n_train else "validation" if rank < n_train + n_val else "test"
        qs[i].split = split
        counts[split] += 1
    db.commit()
    return counts


# --------------------------------------------------------------------------- running
def run_one(db: Session, index: Index, q: EvalQuestion, cfg: dict) -> dict:
    k = cfg["k"]
    t0 = time.perf_counter()
    advanced = cfg.get("transform", "none") != "none" or cfg.get("rerank", "none") != "none" or cfg.get("mmr")
    if advanced:
        ret = pipeline.run(db, index, q.question, mode=cfg["mode"], top_k=k, transform=cfg.get("transform", "none"),
                           mmr_on=bool(cfg.get("mmr")), rerank=cfg.get("rerank", "none"), llm_model=cfg.get("llm_model"))
        hits, llm_tokens = ret["results"], ret["llm_tokens"]
    else:
        ret = retrieval.search(db, index, q.question, cfg["mode"], k)
        hits, llm_tokens = ret["results"], 0
    ms = (time.perf_counter() - t0) * 1000
    ev = q.evidence if q.answerable else []
    m = M.evaluate_question(hits, ev, q.keywords if q.answerable else [], k)
    top1 = hits[0]["contributions"].get("dense", {}).get("score") if hits else None
    out = {"question_id": q.id, "question": q.question, "qtype": q.qtype, "answerable": q.answerable, "split": q.split,
           "metrics": m, "latency_ms": round(ms, 1), "top1_dense_score": top1, "llm_tokens": llm_tokens,
           "top": [{"rank": i + 1, "filename": h["filename"], "pages": h["pages"], "score": h["score"], "chunk_id": h["chunk_id"]}
                   for i, h in enumerate(hits[:k])]}
    if cfg.get("with_answers"):
        out["answer"] = answer_check(db, index, q, cfg)
    return out


def answer_check(db: Session, index: Index, q: EvalQuestion, cfg: dict) -> dict:
    """Generate an answer and run deterministic checks. Optional LLM judge (cfg judge) adds faithfulness and relevance."""
    r = generation.answer(db, index, q.question, model=cfg.get("llm_model"), mode=cfg["mode"], top_k=min(cfg["k"], 8),
                          strict=True, max_output_tokens=int(cfg.get("max_output_tokens", 500)))
    text = r["answer"] or ""
    kws = q.keywords if q.answerable else []
    out = {"abstained": bool(r["abstained"]), "citations_resolve": r["citations"]["all_resolve"] if not r["abstained"] else None,
           "has_citations": r["citations"]["has_citations"] if not r["abstained"] else None,
           "abstention_correct": (r["abstained"] == (not q.answerable)),
           "keyword_hit": (all(M.norm_text(w) in M.norm_text(text) for w in kws) if kws and not r["abstained"] else None),
           "tokens": (r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]) if r["usage"] else 0, "text": text[:600]}
    if cfg.get("judge") and not r["abstained"] and text:
        from ..generation.judge import judge_answer
        try:
            j = judge_answer(q.question, text, [{"n": p["n"], "text": p["text"]} for p in r["context_passages"]], cfg.get("llm_model"))
            out.update(faithfulness=j["faithfulness"], relevance=j["relevance"], contradicted=j["contradicted"],
                       claims_total=j["claims_total"], tokens=out["tokens"] + j["tokens"],
                       unsupported_claims=[c["claim"] for c in j["claims"] if c["verdict"] != "supported"][:5])
        except Exception as e:  # one judge failure must not sink the run; it shows as not judged
            out["judge_error"] = str(e)[:200]
    return out


def summarize(per_q: list[dict], cfg: dict) -> dict:
    scored = [p for p in per_q if p["answerable"]]
    agg = M.aggregate(scored)
    by_type: dict[str, dict] = {}
    for t in sorted({p["qtype"] for p in scored if p["qtype"]}):
        sub = [p for p in scored if p["qtype"] == t]
        by_type[t] = {"n": len(sub), **{m: M.aggregate(sub)[m]["mean"] for m in ("hit", "mrr", "ndcg", "recall")}}
    lat = [p["latency_ms"] for p in per_q]
    un = [p["top1_dense_score"] for p in per_q if not p["answerable"] and p["top1_dense_score"] is not None]
    an = [p["top1_dense_score"] for p in scored if p["top1_dense_score"] is not None]
    gate = None
    if un and an:  # how a minimum-score gate would trade answerable questions kept against unanswerable ones refused
        grid = sorted({round(x, 2) for x in an + un})
        gate = [{"threshold": t, "answerable_kept": round(sum(s >= t for s in an) / len(an), 3),
                 "unanswerable_refused": round(sum(s < t for s in un) / len(un), 3)} for t in grid[::max(1, len(grid) // 12)]]
    ans = [p["answer"] for p in per_q if p.get("answer")]
    answers = None
    if ans:
        def rate(key, rows):
            vals = [a[key] for a in rows if a[key] is not None]
            return {"rate": round(sum(vals) / len(vals), 3) if vals else None, "n": len(vals)}
        answers = {"n": len(ans), "abstention_correct": rate("abstention_correct", ans), "citations_resolve": rate("citations_resolve", ans),
                   "has_citations": rate("has_citations", ans), "keyword_hit": rate("keyword_hit", ans),
                   "tokens": sum(a["tokens"] for a in ans)}
        judged = [a for a in ans if a.get("faithfulness") is not None]
        if judged:
            rel = [a["relevance"] for a in judged if a.get("relevance")]
            answers["faithfulness"] = {"mean": round(sum(a["faithfulness"] for a in judged) / len(judged), 3), "n": len(judged),
                                       "contradicted_claims": sum(a.get("contradicted", 0) for a in judged),
                                       "judge_failures": sum(1 for a in ans if a.get("judge_error"))}
            answers["relevance"] = {"yes": rel.count("yes"), "partial": rel.count("partial"), "no": rel.count("no"), "n": len(rel)}
        answers["not_available"] = ([] if judged else ["faithfulness", "answer relevance"]) + ["answer correctness (needs reference answers)"]
    return {"questions": len(per_q), "answerable": len(scored), "unanswerable": len(per_q) - len(scored), "k": cfg["k"],
            "retrieval": agg, "by_type": by_type,
            "latency_ms": {"p50": M.percentile(lat, 0.5), "p95": M.percentile(lat, 0.95), "mean": round(statistics.fmean(lat), 1) if lat else None},
            "llm_tokens": sum(p["llm_tokens"] for p in per_q),
            "unanswerable_gate": {"top1_dense_unanswerable": {"min": min(un), "max": max(un)} if un else None,
                                  "top1_dense_answerable": {"min": min(an), "median": statistics.median(an)} if an else None,
                                  "sweep": gate},
            "answers": answers,
            "notes": ["Retrieval metrics use only answerable questions that have labelled evidence; others show 'not available'.",
                      "nDCG and MAP credit each labelled evidence item once, at its first rank; the ideal ranking has one chunk per item.",
                      "Confidence intervals are bootstrap intervals and appear only with 10 or more scored questions."]}


# --------------------------------------------------------------------------- comparing
def paired_diff(a: list[float], b: list[float], seed: int = 11) -> dict | None:
    """Mean of b - a over questions scored in both runs, with a paired bootstrap CI (needs >= 10 pairs)."""
    d = [y - x for x, y in zip(a, b)]
    if not d:
        return None
    mean = statistics.fmean(d)
    ci = M.bootstrap_ci(d, seed=seed)
    return {"mean_diff": round(mean, 4), "n": len(d), "ci95": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
            "significant": None if ci is None else (ci[0] > 0 or ci[1] < 0)}


def compare_runs(runs: list[EvalRun]) -> dict:
    if len(runs) < 2:
        raise ValueError("Pick at least two runs to compare.")
    base = runs[0]
    table = []
    for m in M.METRICS:
        table.append({"metric": m, "values": [(r.metrics or {}).get("retrieval", {}).get(m, {}) for r in runs]})
    pq = {r.id: {p["question_id"]: p for p in (r.per_question or [])} for r in runs}
    diffs = []
    for other in runs[1:]:
        shared = [q for q in pq[base.id] if q in pq[other.id] and pq[base.id][q]["answerable"]]
        row = {"baseline": base.id, "other": other.id, "shared_questions": len(shared)}
        for m in ("hit", "mrr", "ndcg", "recall"):
            pairs = [(pq[base.id][q]["metrics"].get(m), pq[other.id][q]["metrics"].get(m)) for q in shared]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            row[m] = paired_diff([x for x, _ in pairs], [y for _, y in pairs])
        changes = []
        for q in shared:
            a, b = pq[base.id][q]["metrics"].get("first_rank"), pq[other.id][q]["metrics"].get("first_rank")
            if a != b:
                better = (b is not None and (a is None or b < a))
                changes.append({"question_id": q, "question": pq[base.id][q]["question"], "baseline_rank": a, "other_rank": b,
                                "direction": "improved" if better else "regressed"})
        row["changed_questions"] = changes
        diffs.append(row)
    return {"runs": [{"id": r.id, "label": r.label, "index": r.index_name, "config": r.config} for r in runs],
            "metrics": table, "paired_vs_first": diffs,
            "note": "Differences are paired per question. 'significant' means the 95% bootstrap interval excludes zero; it is "
                    "null when fewer than 10 paired questions exist."}
