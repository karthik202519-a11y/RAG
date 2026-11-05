"""Chunkers that call a model, measured on real text: semantic (local fastembed, no cost), contextual and proposition (live Azure chat, small samples).

Writes docs/scale/models.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(Path(__file__).parent)]

from corpus import parsed_documents  # noqa: E402
from studio.chunking import run_chunking  # noqa: E402
from studio.chunking.context import use_embedder  # noqa: E402
from studio.embeddings import PROVIDERS  # noqa: E402


def main() -> None:
    out = {}
    docs = parsed_documents(1_000_000, ROOT / "data" / "scale" / "cache")
    prose = [d for d in docs if d["kind"] in ("prose", "docs")]
    # semantic: embed every sentence with the local model, cut where meaning changes
    for label, chars_budget in (("100k", 100_000), ("300k", 300_000)):
        picked, total = [], 0
        for d in prose:
            if total + d["chars"] > chars_budget and picked:
                continue
            picked.append(d)
            total += d["chars"]
        t = time.perf_counter()
        sents = chunks = 0
        with use_embedder(lambda prov, texts: PROVIDERS[prov].embed(texts).vectors):
            for d in picked:
                r = run_chunking(d["pages"], "semantic", {"provider": "fastembed", "min_tokens": 80, "max_tokens": 600})
                chunks += r["stats"]["chunk_count"]
                sents += sum(int(w.split(" sentences")[0].split()[-1]) for w in r["warnings"] if "sentences were embedded" in w)
        secs = time.perf_counter() - t
        out[f"semantic_fastembed_{label}"] = {"chars": total, "documents": len(picked), "sentences_embedded": sents, "chunks": chunks,
                                              "seconds": round(secs, 1), "sentences_per_second": round(sents / max(secs, 1e-9), 1),
                                              "chars_per_second": round(total / max(secs, 1e-9))}
        print(label, out[f"semantic_fastembed_{label}"], flush=True)
    # LLM chunkers: small live samples against the configured chat deployment
    sample = max(prose, key=lambda d: d["chars"])
    for method, params in (("contextual", {"size": 300, "overlap": 0, "max_chunks": 40}), ("proposition", {"source_size": 300, "max_passages": 12})):
        t = time.perf_counter()
        try:
            r = run_chunking(sample["pages"][:30], method, params)
            out[method] = {"document": sample["name"], "chunks": r["stats"]["chunk_count"], "seconds": round(time.perf_counter() - t, 1),
                           "enriched": sum(1 for c in r["chunks"] if c.get("context")) if method == "contextual" else None, "warnings": r["warnings"],
                           "example": (r["chunks"][0].get("context") or r["chunks"][0]["text"])[:240]}
        except Exception as e:
            out[method] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(method, json.dumps(out[method])[:400], flush=True)
    (ROOT / "docs" / "scale").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "scale" / "models.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
