"""Step 6: embed every chunk with text-embedding-3-small.

Usage:  python scripts/06_embed.py <chunks_json> [<chunks_json> ...]

Writes output/embeddings/<name>.npy (rows in the same order as the chunks).
A hash-keyed cache means a text is only ever embedded (and billed) once.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from azure_client import get_client

BATCH = 64
CACHE_FILE = Path(__file__).resolve().parent.parent / "output" / "embeddings" / "cache.jsonl"


def embed_text(chunk: dict) -> str:
    """What the model sees. Text chunks get their section title as context;
    table chunks already start with it."""
    if chunk["type"] == "text" and chunk.get("section"):
        return f"Section: {chunk['section']}\n{chunk['text']}"
    return chunk["text"]


def key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache() -> dict[str, list[float]]:
    if not CACHE_FILE.exists():
        return {}
    with CACHE_FILE.open(encoding="utf-8") as f:
        return {r["k"]: r["v"] for r in map(json.loads, f)}


def embed_many(texts: list[str], cache: dict) -> int:
    """Embed texts missing from the cache; return tokens billed."""
    client, deployment = get_client()
    todo = list(dict.fromkeys(t for t in texts if key(t) not in cache))
    tokens = 0
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            for attempt in range(5):
                try:
                    r = client.embeddings.create(model=deployment, input=batch)
                    break
                except Exception as e:  # rate limit / transient
                    if attempt == 4:
                        raise
                    print(f"  retry {attempt + 1} after {type(e).__name__}")
                    time.sleep(2 ** attempt)
            tokens += r.usage.total_tokens
            for t, d in zip(batch, r.data):
                cache[key(t)] = d.embedding
                f.write(json.dumps({"k": key(t), "v": d.embedding}) + "\n")
            print(f"  embedded {min(i + BATCH, len(todo))}/{len(todo)}")
    return tokens


def main(paths: list[str]) -> None:
    cache = load_cache()
    out_dir = CACHE_FILE.parent
    for p in paths:
        chunks = json.loads(Path(p).read_text(encoding="utf-8"))
        texts = [embed_text(c) for c in chunks]
        print(f"{Path(p).name}: {len(texts)} chunks, {sum(key(t) not in cache for t in texts)} not cached")
        tokens = embed_many(texts, cache)
        matrix = np.array([cache[key(t)] for t in texts], dtype=np.float32)
        # NASA chunks live in output/chunks (kept as chunks_400.npy); any other
        # folder gets its name as a prefix so documents never overwrite each other.
        parent = Path(p).resolve().parent.name
        name = Path(p).stem if parent == "chunks" else f"{parent}_{Path(p).stem}"
        np.save(out_dir / (name + ".npy"), matrix)
        print(f"  saved {matrix.shape}, tokens billed this file: {tokens}")


if __name__ == "__main__":
    main(sys.argv[1:])
