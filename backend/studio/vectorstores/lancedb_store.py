"""LanceDB adapter (embedded, columnar files on disk). Exact scan by default, so recall is 100%; filters are applied before the search."""
import threading
from dataclasses import dataclass

import numpy as np

from ..config import settings

_lock = threading.RLock()
_db = None
_meta: dict[str, dict] = {}


@dataclass
class Hit:
    id: str
    score: float


def status() -> tuple[bool, str]:
    try:
        import lancedb  # noqa: F401
    except ImportError:
        return False, "pip install lancedb"
    return True, "embedded, on disk"


def mode() -> str:
    return "embedded (exact scan), on disk"


def db():
    global _db
    with _lock:
        if _db is None:
            import lancedb
            path = settings.data_dir / "lancedb"
            path.mkdir(parents=True, exist_ok=True)
            _db = lancedb.connect(str(path))
        return _db


def _table(name: str):
    return db().open_table(name)


def _quote(values) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def create_collection(name: str, dim: int, metric: str = "cosine") -> None:
    import pyarrow as pa
    with _lock:
        try:
            db().drop_table(name)
        except Exception:
            pass
        schema = pa.schema([pa.field("id", pa.string()), pa.field("document_id", pa.string()), pa.field("ctype", pa.string()),
                            pa.field("vector", pa.list_(pa.float32(), dim))])
        db().create_table(name, schema=schema)
        _meta[name] = {"dim": dim, "metric": metric}


def _info(name: str) -> dict:
    if name not in _meta:
        t = _table(name)
        dim = t.schema.field("vector").type.list_size
        _meta[name] = {"dim": dim, "metric": "cosine"}
    return _meta[name]


def upsert(name: str, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
    import pyarrow as pa
    t = _table(name)
    rows = pa.table({"id": ids, "document_id": [p.get("document_id", "") for p in payloads], "ctype": [p.get("ctype", "") for p in payloads],
                     "vector": pa.FixedSizeListArray.from_arrays(pa.array(np.asarray(vectors, dtype=np.float32).ravel(), type=pa.float32()), _info(name)["dim"])})
    with _lock:
        t.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)


def search(name: str, vector: np.ndarray, limit: int, document_ids: list[str] | None = None, types: list[str] | None = None) -> list[Hit]:
    t = _table(name)
    metric = _info(name)["metric"]
    q = t.search(np.asarray(vector, dtype=np.float32)).metric({"cosine": "cosine", "dot": "dot", "euclid": "l2"}[metric])
    conds = []
    if document_ids:
        conds.append(f"document_id IN ({_quote(document_ids)})")
    if types:
        conds.append(f"ctype IN ({_quote(types)})")
    if conds:
        q = q.where(" AND ".join(conds), prefilter=True)
    rows = q.limit(limit).to_list()
    out = []
    for r in rows:
        d = float(r["_distance"])
        # cosine distance = 1 - cosine; l2 distance is squared; dot distance is the negated product
        out.append(Hit(id=r["id"], score=-d if metric == "euclid" else 1.0 - d))
    return out


def count(name: str) -> int:
    try:
        return int(_table(name).count_rows())
    except Exception:
        return 0


def collection_dim(name: str) -> int | None:
    try:
        return int(_info(name)["dim"])
    except Exception:
        return None


def delete_collection(name: str) -> None:
    with _lock:
        _meta.pop(name, None)
        try:
            db().drop_table(name)
        except Exception:
            pass


def get_vectors(name: str, ids: list[str]) -> dict[str, np.ndarray]:
    t = _table(name)
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(ids), 500):
        part = t.search().where(f"id IN ({_quote(ids[i:i + 500])})").limit(500).to_list()
        out.update({r["id"]: np.asarray(r["vector"], dtype=np.float32) for r in part})
    return out


def sample_vectors(name: str, limit: int = 1) -> list[list[float]]:
    return [list(map(float, r["vector"])) for r in _table(name).search().limit(limit).to_list()]
