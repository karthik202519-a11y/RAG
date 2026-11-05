"""Vector store adapters. Each module exposes the same functions; `store_for` picks one by the index's `store` column."""
from . import faiss_store, qdrant_store

STORES = {"qdrant": qdrant_store, "faiss": faiss_store}


def store_for(name: str):
    try:
        return STORES[name]
    except KeyError:
        raise ValueError(f"Unknown vector store '{name}'") from None


__all__ = ["qdrant_store", "faiss_store", "STORES", "store_for"]
