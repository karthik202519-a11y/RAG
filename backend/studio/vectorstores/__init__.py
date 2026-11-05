"""Vector store adapters. Each module exposes the same functions; `store_for` picks one by the index's `store` column."""
from . import chroma_store, faiss_store, lancedb_store, pgvector_store, qdrant_store

STORES = {"qdrant": qdrant_store, "faiss": faiss_store, "chroma": chroma_store, "lancedb": lancedb_store, "pgvector": pgvector_store}


def store_for(name: str):
    try:
        return STORES[name]
    except KeyError:
        raise ValueError(f"Unknown vector store '{name}'") from None


__all__ = ["STORES", "store_for", "qdrant_store", "faiss_store", "chroma_store", "lancedb_store", "pgvector_store"]
