"""Embed short query strings, reusing the same hash cache as 06_embed.py."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("embed06", Path(__file__).with_name("06_embed.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def embed_queries(texts: list[str]) -> np.ndarray:
    cache = _m.load_cache()
    _m.embed_many(texts, cache)
    mat = np.array([cache[_m.key(t)] for t in texts], dtype=np.float32)
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)
