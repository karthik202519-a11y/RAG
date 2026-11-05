"""Run every chunking method that needs no model on corpora of 1M, 3M and 10M characters; record time, memory, chunk statistics.

usage: python scripts/scale_bench/bench_chunking.py 1000000 3000000 10000000
Writes docs/scale/chunking_<size>.json
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(Path(__file__).parent)]

from corpus import parsed_documents  # noqa: E402
from studio.chunking import run_chunking  # noqa: E402
from studio.registry import registry  # noqa: E402

SKIP = {"contextual", "proposition", "semantic"}   # need a model; measured separately
SIZE_PARAMS = {"size": 400}


def params_for(method: str) -> dict:
    spec = registry.get(method)
    names = {p.name for p in spec.params}
    out = {k: v for k, v in SIZE_PARAMS.items() if k in names}
    if method == "sentence":
        out["size"] = 400
    return out


def main(sizes: list[int]) -> None:
    out_dir = ROOT / "docs" / "scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    methods = [m.id for m in registry.list("chunking") if m.id not in SKIP]
    for target in sizes:
        t0 = time.perf_counter()
        docs = parsed_documents(target, ROOT / "data" / "scale" / "cache")
        chars = sum(d["chars"] for d in docs)
        kinds = defaultdict(int)
        for d in docs:
            kinds[d["kind"]] += d["chars"]
        print(f"\n=== corpus {target:,}: {len(docs)} docs, {chars:,} chars, built in {time.perf_counter() - t0:.0f}s; mix {dict(kinds)}", flush=True)
        results = {"target_chars": target, "documents": len(docs), "chars": chars, "mix": dict(kinds), "methods": {}}
        for m in methods:
            p = params_for(m)
            rss0 = proc.memory_info().rss
            t = time.perf_counter()
            n = tok = fallback = failed = 0
            errs: list[str] = []
            cov_w = mid = text_chunks = canon_chars = 0
            tokmax = 0
            peak = rss0
            for d in docs:
                try:
                    r = run_chunking(d["pages"], m, p)
                except Exception as e:
                    failed += 1
                    errs.append(f"{d['name']}: {type(e).__name__}: {str(e)[:120]}")
                    continue
                s = r["stats"]
                n += s["chunk_count"]
                tok += s["tokens_total"]
                tokmax = max(tokmax, s["tokens_max"])
                cov_w += s["coverage_ratio"] * s["doc_chars"]
                canon_chars += s["doc_chars"]
                mid += s["mid_sentence_pct"] * s["text_chunks"] / 100
                text_chunks += s["text_chunks"]
                fallback += any("fell back" in w or "fall back" in w.lower() for w in r["warnings"])
                peak = max(peak, proc.memory_info().rss)
            secs = time.perf_counter() - t
            results["methods"][m] = {"seconds": round(secs, 2), "chars_per_second": round(chars / max(secs, 1e-9)), "chunks": n, "tokens": tok,
                                     "mean_tokens": round(tok / max(1, n), 1), "max_tokens": tokmax, "coverage": round(cov_w / max(1, canon_chars), 3),
                                     "mid_sentence_pct": round(100 * mid / max(1, text_chunks), 1), "fell_back_docs": fallback, "failed_docs": failed,
                                     "peak_rss_mb": round(peak / 1e6), "rss_growth_mb": round((peak - rss0) / 1e6), "errors": errs[:3]}
            print(f"{m:22s} {secs:7.1f}s {chars / max(secs, 1e-9) / 1e6:6.2f} Mchar/s  chunks {n:7,}  mean {tok / max(1, n):6.0f} tok  "
                  f"cov {cov_w / max(1, canon_chars):.2f}  fallback {fallback:4d}  failed {failed}  rss {peak / 1e6:.0f}MB" + (f"  ERR {errs[0]}" if errs else ""), flush=True)
        (out_dir / f"chunking_{target}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1_000_000])
