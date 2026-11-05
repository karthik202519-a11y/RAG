"""Run a chunking method on parsed pages: validate parameters, cut, measure, summarize."""
import statistics
import time

from .canon import build_canon
from .methods import METHODS
from .tokens import count

TERMINAL = tuple(".!?:;\"')]”")


class ChunkingError(ValueError):
    """Invalid method or parameters (maps to HTTP 422)."""


def validate_params(method_id: str, raw: dict | None) -> dict:
    """Fill defaults, coerce types, enforce ranges and cross-parameter rules."""
    from ..registry import registry  # lazy: the registry's chunking specs import this package (avoids a cycle)

    spec = registry.get(method_id)
    if spec is None or spec.category != "chunking":
        raise ChunkingError(f"Unknown chunking method '{method_id}'")
    if spec.status == "unavailable":
        raise ChunkingError(f"Method '{method_id}' is not available: {spec.unavailable_reason}")
    if method_id not in METHODS:
        raise ChunkingError(f"Method '{method_id}' is registered but has no implementation")
    raw = dict(raw or {})
    known = {p.name for p in spec.params}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ChunkingError(f"Unknown parameter(s) for {method_id}: {', '.join(unknown)}")
    out: dict = {}
    errors: list[str] = []
    for p in spec.params:
        v = raw.get(p.name, p.default)
        try:
            if p.type == "int":
                if isinstance(v, bool) or (isinstance(v, float) and v != int(v)):
                    raise ValueError
                v = int(v)
            elif p.type == "float":
                v = float(v)
            elif p.type == "bool":
                if not isinstance(v, bool):
                    raise ValueError
            elif p.type == "select":
                if v not in (p.options or []):
                    errors.append(f"{p.label}: must be one of {p.options}")
                    continue
            else:
                v = str(v)
        except (TypeError, ValueError):
            errors.append(f"{p.label}: expected {p.type}")
            continue
        if p.type in ("int", "float"):
            if p.min is not None and v < p.min:
                errors.append(f"{p.label}: minimum is {p.min:g}")
            if p.max is not None and v > p.max:
                errors.append(f"{p.label}: maximum is {p.max:g}")
        out[p.name] = v
    if errors:
        raise ChunkingError("; ".join(errors))
    if "overlap" in out and "size" in out and out["overlap"] >= out["size"]:
        raise ChunkingError(f"Overlap ({out['overlap']}) must be smaller than chunk size ({out['size']}).")
    if "child_overlap" in out and "child_size" in out and out["child_overlap"] >= out["child_size"]:
        raise ChunkingError(f"Child overlap ({out['child_overlap']}) must be smaller than child size ({out['child_size']}).")
    if "parent_size" in out and "child_size" in out and out["child_size"] >= out["parent_size"]:
        raise ChunkingError(f"Child size ({out['child_size']}) must be smaller than parent size ({out['parent_size']}).")
    if "min_tokens" in out and "max_tokens" in out and out["min_tokens"] > out["max_tokens"]:
        raise ChunkingError(f"Minimum chunk size ({out['min_tokens']}) must not exceed the maximum ({out['max_tokens']}).")
    if "stride" in out and "size" in out and out["stride"] > out["size"]:
        raise ChunkingError(f"Stride ({out['stride']}) larger than window size ({out['size']}) would skip text.")
    return out


def _pct(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def run_chunking(pages: list[dict], method_id: str, raw_params: dict | None,
                 price_per_million_tokens: float | None = None) -> dict:
    """Return {"method", "params", "chunks", "stats", "warnings"}; chunks carry real character spans."""
    params = validate_params(method_id, raw_params)
    tokenizer = params.get("tokenizer", "cl100k_base")
    canon = build_canon(pages)
    t0 = time.perf_counter()
    try:
        cuts, warnings = METHODS[method_id](canon, params)
    except ValueError as e:
        raise ChunkingError(str(e)) from e
    elapsed = time.perf_counter() - t0
    if not canon.text.strip():
        raise ChunkingError("This document has no extractable text.")

    chunks: list[dict] = []
    prev_end: int | None = None
    for cut in cuts:
        s, e = cut.get("start"), cut.get("end")
        located = s is not None and e is not None
        text = cut.get("text") or canon.text[s:e]
        if not text.strip():
            continue
        if located:
            pages_ = canon.pages_for(s, e)
            kinds = canon.kinds_in(s, e)
            ctype = cut.get("type") or ("table" if kinds == {"table"} else "text")
            overlap = max(0, prev_end - s) if prev_end is not None else 0
            prev_end = max(prev_end or 0, e)
        else:
            pages_ = sorted(set(cut.get("pages", [])))
            ctype = cut.get("type", "text")
            overlap = 0
        chunks.append({
            "index": len(chunks),
            "start": s if located else None,
            "end": e if located else None,
            "text": text,
            "type": ctype,
            "chars": len(text),
            "tokens": count(text, tokenizer),
            "pages": pages_,
            "printed_pages": [canon.printed.get(p) for p in pages_],
            "section": canon.section_at(s, e) if located else cut.get("section"),
            "parent_start": cut.get("parent", {}).get("start"),
            "parent_end": cut.get("parent", {}).get("end"),
            "overlap_prev_chars": overlap,
            "ends_mid_sentence": ctype == "text" and not text.rstrip().endswith(TERMINAL),
        })

    tokens = sorted(c["tokens"] for c in chunks)
    chars_total = sum(c["chars"] for c in chunks)
    covered = _covered(chunks, len(canon.text))
    stats = {
        "chunk_count": len(chunks),
        "text_chunks": sum(c["type"] == "text" for c in chunks),
        "table_chunks": sum(c["type"] == "table" for c in chunks),
        "doc_chars": len(canon.text),
        "tokens_total": sum(tokens),
        "tokens_mean": round(statistics.fmean(tokens), 1) if tokens else 0,
        "tokens_median": int(statistics.median(tokens)) if tokens else 0,
        "tokens_p10": _pct(tokens, 0.10), "tokens_p90": _pct(tokens, 0.90),
        "tokens_min": tokens[0] if tokens else 0, "tokens_max": tokens[-1] if tokens else 0,
        "histogram": _histogram(tokens),
        "duplication_ratio": round(chars_total / max(1, len(canon.text)), 3),  # 1.0 = no repeated text
        "coverage_ratio": round(covered, 3),
        "mid_sentence_pct": round(100 * sum(c["ends_mid_sentence"] for c in chunks) / max(1, sum(c["type"] == "text" for c in chunks)), 1),
        "estimated_embedding_tokens": sum(tokens),
        "estimated_embedding_cost_usd": (round(sum(tokens) / 1e6 * price_per_million_tokens, 6)
                                         if price_per_million_tokens is not None else None),
        "vectors_to_store": len(chunks),
        "parents": len({(c["parent_start"], c["parent_end"]) for c in chunks if c["parent_start"] is not None}),
        "processing_seconds": round(elapsed, 4),
    }
    return {"method": method_id, "params": params, "chunks": chunks, "stats": stats, "warnings": warnings,
            "doc_text": canon.text}


def _covered(chunks: list[dict], n: int) -> float:
    spans = sorted((c["start"], c["end"]) for c in chunks if c["start"] is not None)
    total, cur_e = 0, 0
    for s, e in spans:
        s = max(s, cur_e)
        if e > s:
            total += e - s
            cur_e = e
    return total / n if n else 0.0


def _histogram(tokens: list[int], bins: int = 10) -> list[dict]:
    if not tokens:
        return []
    lo, hi = tokens[0], tokens[-1]
    if lo == hi:
        return [{"from": lo, "to": hi, "count": len(tokens)}]
    width = max(1, -(-(hi - lo + 1) // bins))
    out = [{"from": lo + i * width, "to": lo + (i + 1) * width - 1, "count": 0} for i in range(bins)]
    for t in tokens:
        out[min(bins - 1, (t - lo) // width)]["count"] += 1
    return out
