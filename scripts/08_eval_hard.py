"""Step 8: harder retrieval test over BOTH documents in one index (400-token chunks).

Usage:  python scripts/08_eval_hard.py

For each question we report
  rank      : rank of the first chunk on a correct (doc, page); every page group
              must be covered inside the top K for a multi-hop question to pass
  evidence  : do the top-K chunks contain the answer keywords?
  on_page   : do the keywords exist in the extracted text of the correct page?
              (False here = extraction lost the answer, retrieval cannot help)
  top1_sim  : similarity of the best chunk (used to look at unanswerable questions)
"""
import json
import os
from pathlib import Path

import numpy as np

from embed_utils import embed_queries

ROOT = Path(__file__).resolve().parent.parent
K = 5
V = os.environ.get("RAG_VARIANT", "")  # "" = first chunker, "_v2" = after clean-up
INDEX = {
    "nasa": (f"output/chunks{V}/chunks_400.json",
             f"output/embeddings/chunks{V}_chunks_400.npy" if V else "output/embeddings/chunks_400.npy",
             "output/extracted/nasa_systems_engineering_handbook_0.json"),
    "ipcc": (f"output/chunks_ipcc{V}/chunks_400.json", f"output/embeddings/chunks_ipcc{V}_chunks_400.npy",
             "output/extracted_ipcc/IPCC_AR6_SYR_FullVolume.json"),
}


def norm(text: str) -> str:
    """Lower-case and collapse all whitespace so line breaks / double spaces
    inside a chunk can't hide a keyword."""
    return " ".join(text.lower().split())


def load():
    chunks, mats, page_text = [], [], {}
    for doc, (c, e, x) in INDEX.items():
        cs = json.loads((ROOT / c).read_text(encoding="utf-8"))
        for ch in cs:
            ch["source"] = doc
        chunks += cs
        m = np.load(ROOT / e)
        assert len(m) == len(cs), f"{doc}: {len(m)} vectors vs {len(cs)} chunks"
        mats.append(m / np.linalg.norm(m, axis=1, keepdims=True))
        for p in json.loads((ROOT / x).read_text(encoding="utf-8")):
            page_text[(doc, p["pdf_page"])] = " ".join(
                i.get("text", "") + i.get("markdown", "") for i in p["items"]).lower()
    return chunks, np.vstack(mats), page_text


def main() -> None:
    chunks, mat, page_text = load()
    qs = json.loads((ROOT / "eval/questions_hard.json").read_text(encoding="utf-8"))
    sims = embed_queries([q["question"] for q in qs]) @ mat.T
    rows = []
    for q, s in zip(qs, sims):
        top = np.argsort(-s)[:K]
        tops = [chunks[i] for i in top]
        ranks = []
        for group in q["page_groups"]:
            ranks.append(next((r + 1 for r, c in enumerate(tops)
                               if c["source"] == q["doc"] and set(c["pages"]) & set(group)), None))
        hit = bool(q["page_groups"]) and all(ranks)
        joined = norm(" ".join(c["text"] for c in tops))
        kws = q["keywords"]
        evidence = all(norm(k) in joined for k in kws) if kws else None
        # each keyword must exist in the extracted text of SOME correct page
        correct_pages = [p for g in q["page_groups"] for p in g]
        on_page = (all(any(norm(k) in norm(page_text.get((q["doc"], p), "")) for p in correct_pages)
                       for k in kws) if kws else None)
        rows.append({"id": q["id"], "cat": q["cat"], "hit": hit, "ranks": ranks, "evidence": evidence,
                     "on_page": on_page, "top1_sim": round(float(s[top[0]]), 3),
                     "top": [(c["source"], c["pages"], c["type"]) for c in tops],
                     "image": q.get("needs_image", False)})

    print(f"{'id':<4}{'category':<17}{'hit@5':<7}{'ranks':<10}{'evidence':<10}{'on_page':<9}{'sim':<7}top-1")
    for r in rows:
        t = r["top"][0]
        print(f"{r['id']:<4}{r['cat']:<17}{str(r['hit']):<7}{str(r['ranks']):<10}{str(r['evidence']):<10}"
              f"{str(r['on_page']):<9}{r['top1_sim']:<7}{t[0]} p{t[1]} {t[2]}"
              + ("  [needs image]" if r["image"] else ""))
    ans = [r for r in rows if r["cat"] != "unanswerable"]
    print(f"\nAnswerable: hit@5 {sum(r['hit'] for r in ans)}/{len(ans)}")
    print("Unanswerable top-1 sims:", [r["top1_sim"] for r in rows if r["cat"] == "unanswerable"],
          "| answerable top-1 sims min/median:",
          min(r["top1_sim"] for r in ans), sorted(r["top1_sim"] for r in ans)[len(ans) // 2])
    out = ROOT / "output/eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"hard_results{V}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
