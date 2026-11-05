"""Retrieval metrics. Pure functions, hand-checkable.

Relevance is judged against labelled evidence: a retrieved chunk is relevant if it comes from the
labelled document and its pages overlap the labelled pages. A question may have several evidence items
(multi-hop); recall counts how many of them appear in the top-k.

nDCG uses binary relevance and an ideal ranking that places one chunk per evidence item first, because the
total number of relevant chunks in an index is unknown. Each evidence item is credited once, at its first rank
(so nDCG and average precision never exceed 1); precision still counts every relevant chunk. Metrics that cannot be computed return None
(shown as "not available"), never zero.
"""
import math
import random
import re
import statistics

Evidence = dict  # {"document": str, "pages": [int, ...]}


def doc_matches(spec: str | None, filename: str, document_id: str) -> bool:
    if not spec:
        return True
    return spec == document_id or spec.lower() in filename.lower()


def item_matches(item: Evidence, hit: dict) -> bool:
    if not doc_matches(item.get("document"), hit["filename"], hit["document_id"]):
        return False
    pages = item.get("pages") or []
    return (not pages) or bool(set(pages) & set(hit.get("pages") or []))


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def evaluate_question(hits: list[dict], evidence: list[Evidence], keywords: list[str], k: int) -> dict:
    """Metrics for one question given ranked hits (best first). Evidence may be empty (unanswerable)."""
    top = hits[:k]
    out: dict = {"returned": len(hits)}
    kw_text = norm_text(" ".join(h["text"] for h in top))
    out["keyword_coverage"] = (sum(norm_text(w) in kw_text for w in keywords) / len(keywords)) if keywords else None
    if not evidence:
        for m in ("hit", "recall", "precision", "mrr", "ndcg", "ap"):
            out[m] = None
        out["first_rank"] = None
        return out

    rel = [any(item_matches(it, h) for it in evidence) for h in top]
    item_rank = []
    for it in evidence:
        item_rank.append(next((i + 1 for i, h in enumerate(top) if item_matches(it, h)), None))
    found = [r for r in item_rank if r]
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    out["first_rank"] = first
    out["item_ranks"] = item_rank
    out["hit"] = 1.0 if first else 0.0
    out["recall"] = len(found) / len(evidence)
    out["precision"] = (sum(rel) / len(top)) if top else 0.0
    out["mrr"] = 1.0 / first if first else 0.0
    # Each evidence item earns credit once, at the first rank where it appears. Several chunks from the same labelled
    # page must not earn extra credit, or nDCG and AP could exceed 1 (the ideal ranking has one chunk per item).
    ranks = sorted(found)
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(evidence), k)))
    out["ndcg"] = dcg / ideal if ideal else 0.0
    out["ap"] = sum((j + 1) / r for j, r in enumerate(ranks)) / min(len(evidence), k)
    return out


def mean_defined(values: list[float | None]) -> tuple[float | None, int]:
    v = [x for x in values if x is not None]
    return (statistics.fmean(v) if v else None), len(v)


def bootstrap_ci(values: list[float], iters: int = 1000, seed: int = 7, level: float = 0.95) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval of the mean. Needs at least 10 samples; otherwise None."""
    if len(values) < 10:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(iters))
    lo = means[int((1 - level) / 2 * iters)]
    hi = means[min(iters - 1, int((1 + level) / 2 * iters))]
    return lo, hi


METRICS = ("hit", "recall", "precision", "mrr", "ndcg", "ap", "keyword_coverage")


def aggregate(per_question: list[dict]) -> dict:
    """Mean of each metric over the questions where it is defined, with n and a bootstrap CI when possible."""
    out = {}
    for m in METRICS:
        vals = [q["metrics"][m] for q in per_question if q["metrics"].get(m) is not None]
        mean = statistics.fmean(vals) if vals else None
        ci = bootstrap_ci(vals) if vals else None
        out[m] = {"mean": None if mean is None else round(mean, 4), "n": len(vals),
                  "ci95": None if ci is None else [round(ci[0], 4), round(ci[1], 4)]}
    return out


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]
