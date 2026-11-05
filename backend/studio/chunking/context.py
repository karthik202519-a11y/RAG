"""Carries an embedding function into chunkers that need one (semantic chunking), without making every
chunker take a database session. The caller sets it for the duration of the call."""
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

Embedder = Callable[[str, list[str]], np.ndarray]  # (provider id, texts) -> L2-normalised vectors
_embedder: ContextVar[Embedder | None] = ContextVar("chunk_embedder", default=None)


@contextmanager
def use_embedder(fn: Embedder) -> Iterator[None]:
    token = _embedder.set(fn)
    try:
        yield
    finally:
        _embedder.reset(token)


def get_embedder() -> Embedder | None:
    return _embedder.get()
