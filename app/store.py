"""On-disk document store. One folder per document under DATA_DIR/documents/<doc_id>/:

    source.pdf            the uploaded file
    meta.json             filename, hash, counts, which chunk sizes / embeddings exist
    pages.json            extracted pages (items, printed page, section, needs_vision)
    visual_pages/*.png    renders of diagram/chart pages
    chunks_<size>.json    chunks for a given target size (tokens)
    vectors_<size>.npy    one normalized embedding per chunk, same order
"""
import hashlib
import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config

DOC_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class NotFound(KeyError):
    pass


class Store:
    def __init__(self, root: Path | None = None) -> None:
        self._root_override = root
        self._lock = threading.Lock()
        self._vec_cache: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------ paths
    @property
    def root(self) -> Path:
        return (self._root_override or config.DATA_DIR) / "documents"

    def doc_dir(self, doc_id: str) -> Path:
        if not DOC_ID_RE.match(doc_id):
            raise NotFound(doc_id)
        return self.root / doc_id

    def _require(self, doc_id: str) -> Path:
        d = self.doc_dir(doc_id)
        if not (d / "meta.json").exists():
            raise NotFound(doc_id)
        return d

    # -------------------------------------------------------------- documents
    def create(self, filename: str, content: bytes) -> tuple[dict, bool]:
        """Save an upload. Returns (meta, created). Same bytes -> same id, not re-saved."""
        digest = hashlib.sha256(content).hexdigest()
        doc_id = digest[:12]
        d = self.root / doc_id
        with self._lock:
            if (d / "meta.json").exists():
                return self.get(doc_id), False
            d.mkdir(parents=True, exist_ok=True)
            (d / "source.pdf").write_bytes(content)
            meta = {
                "doc_id": doc_id,
                "filename": Path(filename).name or "document.pdf",
                "sha256": digest,
                "size_bytes": len(content),
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "extracted": False,
                "pages": None,
                "tables": None,
                "text_blocks": None,
                "needs_vision_pages": None,
                "chunked": {},   # size -> {"chunks", "text", "table"}
                "embedded": {},  # size -> {"chunks", "dim", "backend", "tokens_billed"}
            }
            self.save_meta(doc_id, meta)
        return meta, True

    def get(self, doc_id: str) -> dict:
        return json.loads((self._require(doc_id) / "meta.json").read_text(encoding="utf-8"))

    def save_meta(self, doc_id: str, meta: dict) -> None:
        d = self.doc_dir(doc_id)
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        tmp.replace(d / "meta.json")

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            if (d / "meta.json").exists():
                out.append(json.loads((d / "meta.json").read_text(encoding="utf-8")))
        return sorted(out, key=lambda m: m["uploaded_at"])

    def delete(self, doc_id: str) -> None:
        d = self._require(doc_id)
        with self._lock:
            shutil.rmtree(d)
            self._vec_cache = {k: v for k, v in self._vec_cache.items() if k[0] != doc_id}

    # ------------------------------------------------------------ pages/chunks
    def source_path(self, doc_id: str) -> Path:
        return self._require(doc_id) / "source.pdf"

    def save_pages(self, doc_id: str, pages: list[dict]) -> None:
        (self._require(doc_id) / "pages.json").write_text(
            json.dumps(pages, ensure_ascii=False), encoding="utf-8")

    def load_pages(self, doc_id: str) -> list[dict]:
        p = self._require(doc_id) / "pages.json"
        if not p.exists():
            raise FileNotFoundError("not extracted")
        return json.loads(p.read_text(encoding="utf-8"))

    def save_chunks(self, doc_id: str, size: int, chunks: list[dict]) -> None:
        (self._require(doc_id) / f"chunks_{size}.json").write_text(
            json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    def load_chunks(self, doc_id: str, size: int) -> list[dict]:
        p = self._require(doc_id) / f"chunks_{size}.json"
        if not p.exists():
            raise FileNotFoundError(f"not chunked at size {size}")
        return json.loads(p.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- vectors
    def save_vectors(self, doc_id: str, size: int, matrix: np.ndarray) -> None:
        np.save(self._require(doc_id) / f"vectors_{size}.npy", matrix.astype(np.float32))
        self._vec_cache = {k: v for k, v in self._vec_cache.items() if k[:2] != (doc_id, size)}

    def load_vectors(self, doc_id: str, size: int) -> np.ndarray:
        p = self._require(doc_id) / f"vectors_{size}.npy"
        if not p.exists():
            raise FileNotFoundError(f"not embedded at size {size}")
        k = (doc_id, size, p.stat().st_mtime_ns)
        if k not in self._vec_cache:
            self._vec_cache[k] = np.load(p)
        return self._vec_cache[k]
