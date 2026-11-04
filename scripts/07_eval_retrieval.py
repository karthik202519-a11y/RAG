"""Step 7: score retrieval at each chunk size against eval/questions.json.

Usage:  python scripts/07_eval_retrieval.py [size ...]      (default: 200 400 800)

For every question: embed it, take the top-K chunks by cosine similarity, and
check whether any retrieved chunk lies on a correct page ("hit").
  hit@k : fraction of questions whose correct page is within the top k chunks
  MRR   : mean of 1/rank of the first correct chunk (0 if not in top K)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from azure_client import get_client  # noqa: F401  (config check)
from embed_utils import embed_queries

ROOT = Path(__file__).resolve().parent.parent
K = 5


def evaluate(size: int, questions: list[dict], qvecs: np.ndarray) -> dict:
    v = os.environ.get("RAG_VARIANT", "")  # "" = first chunker, "_v2" = after clean-up
    chunks = json.loads((ROOT / f"output/chunks{v}/chunks_{size}.json").read_text(encoding="utf-8"))
    mat = np.load(ROOT / (f"output/embeddings/chunks{v}_chunks_{size}.npy" if v
                          else f"output/embeddings/chunks_{size}.npy"))
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    sims = qvecs @ mat.T
    rows = []
    for q, s in zip(questions, sims):
        top = np.argsort(-s)[:K]
        rank = next((r + 1 for r, i in enumerate(top) if set(chunks[i]["pages"]) & set(q["pdf_pages"])), None)
        rows.append({"id": q["id"], "type": q["type"], "rank": rank,
                     "top": [(int(i), chunks[i]["type"], chunks[i]["pages"], round(float(s[i]), 3)) for i in top]})
    return {"size": size, "rows": rows}


def summarize(res: dict) -> str:
    rows = res["rows"]

    def agg(sub):
        n = len(sub)
        h = lambda k: sum(1 for r in sub if r["rank"] and r["rank"] <= k)
        mrr = sum(1 / r["rank"] for r in sub if r["rank"]) / n
        return f"{h(1)}/{n}", f"{h(3)}/{n}", f"{h(5)}/{n}", f"{mrr:.2f}"

    line = f"{res['size']:>5} | all   | " + " | ".join(agg(rows))
    for t in ("prose", "table", "diagram"):
        sub = [r for r in rows if r["type"] == t]
        if sub:
            line += f"\n      | {t:<5} | " + " | ".join(agg(sub))
    return line


def main(sizes: list[int]) -> None:
    questions = json.loads((ROOT / "eval/questions.json").read_text(encoding="utf-8"))
    qvecs = embed_queries([q["question"] for q in questions])
    results = [evaluate(s, questions, qvecs) for s in sizes]

    print("size  | slice | hit@1 | hit@3 | hit@5 | MRR")
    for r in results:
        print(summarize(r))
    out = ROOT / "output/eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results{os.environ.get('RAG_VARIANT', '')}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print("\nQuestions missed in the top", K, "(size: question ids):")
    for r in results:
        print(f"  {r['size']}: {[x['id'] for x in r['rows'] if not x['rank']]}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [200, 400, 800])
