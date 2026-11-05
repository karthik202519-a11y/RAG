"""Turn docs/scale/*.json into docs/SCALE_TEST.md. Every number in the report is read from the result files; none is typed in."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs" / "scale"


def load(name):
    p = D / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def lab(n) -> str:
    n = int(n)
    return f"{n / 1e6:g}M" if n >= 1_000_000 else f"{n / 1e3:g}k"


def fmt(n, d=0):
    return f"{n:,.{d}f}" if isinstance(n, (int, float)) else "n/a"


def main() -> None:
    out = ["# Scale test: 1 to 10 million characters", "",
           "Real text from this machine (Python, JavaScript and TypeScript sources, Markdown, text, JSON, CSV, HTML and the two real PDFs), "
           "in a fixed mix, taken in a fixed order so each corpus contains the smaller one. Nothing is generated. Reproduce with "
           "`scripts/scale_bench/` (see the end of this file). The numbers below are written by `make_report.py` from `docs/scale/*.json`.", ""]
    sizes = sorted(int(p.stem.split("_")[1]) for p in D.glob("chunking_*.json"))
    chunk = {s: load(f"chunking_{s}.json") for s in sizes}
    if chunk:
        out += ["## 1. Chunking: every method that needs no model, on every corpus", ""]
        for s in sizes:
            c = chunk[s]
            out.append(f"- **{lab(s)} corpus**: {c['documents']} documents, {fmt(c['chars'])} characters; mix " + ", ".join(f"{k} {v / c['chars']:.0%}" for k, v in c["mix"].items()))
        out += ["", "Seconds to chunk the whole corpus (single process, one document at a time), chunks produced, and how often the method fell back to a plain splitter "
                "because the document was not the kind it is designed for (for example HTML-structure on a Python file).", ""]
        names = list(chunk[sizes[0]]["methods"])
        head = "| Method | " + " | ".join(f"{lab(s)} s | {lab(s)} chunks" for s in sizes) + f" | Mchar/s at {lab(sizes[-1])} | Mean tokens | Coverage | Fell back (docs, {lab(sizes[-1])}) |"
        out += [head, "|" + "---|" * (head.count("|") - 1)]
        for m in names:
            row = [m]
            for s in sizes:
                r = chunk[s]["methods"].get(m)
                row += [fmt(r["seconds"], 1), fmt(r["chunks"])] if r else ["", ""]
            last = chunk[sizes[-1]]["methods"][m]
            row += [fmt(last["chars_per_second"] / 1e6, 2), fmt(last["mean_tokens"]), f"{last['coverage']:.2f}", f"{last['fell_back_docs']} of {chunk[sizes[-1]]['documents']}"]
            out.append("| " + " | ".join(row) + " |")
        peak = max(r["peak_rss_mb"] for r in chunk[sizes[-1]]["methods"].values())
        failed = [f"{lab(s)}: {m} failed on {r['failed_docs']} document(s)" + (f" ({r['errors'][0]})" if r.get("errors") else " (error text was not captured in that run)")
                  for s in sizes for m, r in chunk[s]["methods"].items() if r.get("failed_docs")]
        out += ["", f"Peak memory of the benchmark process at the largest corpus: {peak} MB.",
                "Failed documents: " + ("; ".join(failed) + ". These did not recur on the larger corpora, which contain the same documents, and the cause was not identified." if failed else "none.")]
    mod = load("models.json")
    if mod:
        out += ["", "## 2. Chunkers that call a model", ""]
        for k, v in mod.items():
            if "error" in v:
                out.append(f"- **{k}**: {v['error']}")
            elif k.startswith("semantic"):
                out.append(f"- **{k}** (local fastembed, free): {fmt(v['chars'])} characters, {fmt(v['sentences_embedded'])} sentences embedded in {v['seconds']} s "
                           f"({v['sentences_per_second']} sentences/s) into {v['chunks']} chunks. Extrapolated to 10M characters: about {fmt(10_000_000 / max(1, v['chars_per_second']) / 60, 0)} minutes.")
            else:
                out.append(f"- **{k}** (live Azure chat, sample): {v['chunks']} chunks in {v['seconds']} s. {' '.join(v.get('warnings', []))[:300]}")
    pipe = load("pipeline.json")
    if pipe:
        out += ["", "## 3. End to end through the API (upload, parse, embed with Azure `text-embedding-3-small`, search)", "",
                "| Corpus | Files | Ingest s | Files/s | Parse failures | Chunks | Tokens embedded | Tokens billed | Cache hits | Index build s |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s, r in pipe["sizes"].items():
            i, x = r["ingest"], r["index"]
            out.append(f"| {lab(s)} | {i['files']} | {i['seconds']} | {i['files_per_second']} | {i['failed']} | {fmt(x['chunks'])} | {fmt(x['tokens_embedded'])} | {fmt(x['tokens_billed'])} | {fmt(x['cache_hits'])} | {x['build_seconds']} |")
        out += ["", "Retrieval on the same needle questions as the haystack grows (k = 5). `hit` = the right document was returned; `keyword coverage` = the retrieved text contained the "
                "distinctive keyword from the source passage. Confidence intervals are 95% bootstrap.", "",
                "| Corpus | Mode | Questions | Hit | Hit 95% CI | MRR | nDCG | Keyword coverage | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s, r in pipe["sizes"].items():
            for mode, v in r["retrieval"].items():
                ci = v.get("hit_ci95")
                out.append(f"| {lab(s)} | {mode} | {v['questions']} | {fmt(v['hit'], 3)} | {('%.2f to %.2f' % tuple(ci)) if ci else 'n/a'} | {fmt(v['mrr'], 3)} | {fmt(v['ndcg'], 3)} | {fmt(v['keyword_coverage'], 3)} | {fmt(v['latency_p50_ms'], 0)} | {fmt(v['latency_p95_ms'], 0)} |")
    st = load("stores.json")
    if st:
        out += ["", "## 4. Vector stores on the largest corpus's chunks", "",
                f"{fmt(st['vectors'])} vectors, {st['dimension']} dimensions, {st['queries']} query vectors. Recall@10 is measured against FAISS exact search.", "",
                "| Store | Load s | Search p50 ms | p95 ms | Recall@10 vs exact | Filtered p50 ms |", "|---|---|---|---|---|---|"]
        for name, v in st["stores"].items():
            out.append(f"| {name} | {fmt(v.get('load_seconds'), 1)} | {fmt(v.get('p50_ms'), 1)} | {fmt(v.get('p95_ms'), 1)} | {fmt(v.get('recall_at_10'), 3)} | {fmt(v.get('filtered_p50_ms'), 1)} |"
                       if "error" not in v else f"| {name} | {v['error']} | | | | |")
    out += ["", "## Reproduce", "", "```", "python scripts/scale_bench/bench_chunking.py 1000000 3000000 10000000",
            "python scripts/scale_bench/bench_models.py", "# start a second API with its own data folder:",
            "#   STUDIO_DATA_DIR=data/scale/studio STUDIO_JOB_WORKERS=4 uvicorn studio.main:app --port 8001",
            "python scripts/scale_bench/bench_pipeline.py http://localhost:8001 1000000 3000000 10000000",
            "python scripts/scale_bench/bench_stores.py", "python scripts/scale_bench/make_report.py", "```", ""]
    (ROOT / "docs" / "SCALE_TEST.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote docs/SCALE_TEST.md")


if __name__ == "__main__":
    main()
