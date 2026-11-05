"""pgvector adapter: vectors live in a PostgreSQL table with an HNSW index. Needs PGVECTOR_URL pointing at a database where the
`vector` extension can be created (for example the pgvector/pgvector Docker image)."""
import re
import threading
from dataclasses import dataclass

import numpy as np

from ..config import settings

_lock = threading.RLock()
_engine = None
OPS = {"cosine": ("vector_cosine_ops", "<=>"), "dot": ("vector_ip_ops", "<#>"), "euclid": ("vector_l2_ops", "<->")}
_metric: dict[str, str] = {}


@dataclass
class Hit:
    id: str
    score: float


def _url() -> str:
    u = settings.pgvector_url
    return u.replace("postgresql://", "postgresql+psycopg://", 1) if u.startswith("postgresql://") else u


def status() -> tuple[bool, str]:
    if not settings.pgvector_url:
        return False, "Set in .env: PGVECTOR_URL (a PostgreSQL with the pgvector extension, e.g. docker image pgvector/pgvector:pg16)"
    try:
        with engine().connect() as c:
            from sqlalchemy import text
            ok = c.execute(text("SELECT 1 FROM pg_available_extensions WHERE name='vector'")).first()
        return (True, "PostgreSQL with pgvector") if ok else (False, "Connected, but this PostgreSQL does not ship the pgvector extension")
    except Exception as e:
        return False, f"Cannot connect to PGVECTOR_URL: {type(e).__name__}"


def mode() -> str:
    return "PostgreSQL + pgvector (HNSW)"


def engine():
    global _engine
    with _lock:
        if _engine is None:
            from sqlalchemy import create_engine
            _engine = create_engine(_url(), pool_size=5, pool_pre_ping=True)
        return _engine


def _t(name: str) -> str:
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,60}", name):
        raise ValueError(f"Unsafe collection name '{name}'")
    return f"vs_{name}"


def _vec(v) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in v) + "]"


def create_collection(name: str, dim: int, metric: str = "cosine") -> None:
    from sqlalchemy import text
    ops, _ = OPS[metric]
    t = _t(name)
    with engine().begin() as c:
        c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        c.execute(text(f"DROP TABLE IF EXISTS {t}"))
        c.execute(text(f"CREATE TABLE {t} (id char(32) PRIMARY KEY, document_id text, ctype text, embedding vector({int(dim)}))"))
        c.execute(text(f"COMMENT ON TABLE {t} IS 'metric={metric};dim={int(dim)}'"))
        if dim <= 2000:  # HNSW supports up to 2000 dimensions; larger vectors are scanned exactly
            c.execute(text(f"CREATE INDEX ON {t} USING hnsw (embedding {ops})"))
    _metric[name] = metric


def _metric_of(name: str) -> str:
    if name not in _metric:
        from sqlalchemy import text
        with engine().connect() as c:
            cm = c.execute(text("SELECT obj_description(to_regclass(:t)::oid)"), {"t": _t(name)}).scalar() or ""
        m = re.search(r"metric=(\w+)", cm)
        _metric[name] = m.group(1) if m else "cosine"
    return _metric[name]


def upsert(name: str, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
    from sqlalchemy import text
    t = _t(name)
    rows = [{"i": i, "d": p.get("document_id", ""), "c": p.get("ctype", ""), "v": _vec(v)} for i, v, p in zip(ids, vectors, payloads)]
    with engine().begin() as c:
        c.execute(text(f"INSERT INTO {t} (id, document_id, ctype, embedding) VALUES (:i, :d, :c, CAST(:v AS vector)) "
                       f"ON CONFLICT (id) DO UPDATE SET document_id = EXCLUDED.document_id, ctype = EXCLUDED.ctype, embedding = EXCLUDED.embedding"), rows)


def search(name: str, vector: np.ndarray, limit: int, document_ids: list[str] | None = None, types: list[str] | None = None) -> list[Hit]:
    from sqlalchemy import bindparam, text
    t, metric = _t(name), _metric_of(name)
    op = OPS[metric][1]
    where, params = [], {"q": _vec(vector), "k": int(limit)}
    if document_ids:
        where.append("document_id IN :docs")
        params["docs"] = list(document_ids)
    if types:
        where.append("ctype IN :types")
        params["types"] = list(types)
    sql = text(f"SELECT id, embedding {op} CAST(:q AS vector) AS d FROM {t}" + (" WHERE " + " AND ".join(where) if where else "")
               + f" ORDER BY embedding {op} CAST(:q AS vector) LIMIT :k")
    sql = sql.bindparams(*[bindparam(n, expanding=True) for n in ("docs", "types") if n in params])
    with engine().begin() as c:
        c.execute(text("SET LOCAL hnsw.ef_search = 200"))  # recall at the cost of a little time; the default (40) is low
        rows = c.execute(sql, params).all()
    # pgvector returns distances: cosine distance, negative inner product, or L2 distance
    return [Hit(id=r[0].strip(), score=float(-r[1]) if metric in ("dot", "euclid") else float(1.0 - r[1])) for r in rows]


def count(name: str) -> int:
    from sqlalchemy import text
    try:
        with engine().connect() as c:
            return int(c.execute(text(f"SELECT count(*) FROM {_t(name)}")).scalar() or 0)
    except Exception:
        return 0


def collection_dim(name: str) -> int | None:
    from sqlalchemy import text
    try:
        with engine().connect() as c:
            cm = c.execute(text("SELECT obj_description(to_regclass(:t)::oid)"), {"t": _t(name)}).scalar() or ""
        m = re.search(r"dim=(\d+)", cm)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def delete_collection(name: str) -> None:
    from sqlalchemy import text
    _metric.pop(name, None)
    with engine().begin() as c:
        c.execute(text(f"DROP TABLE IF EXISTS {_t(name)}"))


def get_vectors(name: str, ids: list[str]) -> dict[str, np.ndarray]:
    from sqlalchemy import bindparam, text
    sql = text(f"SELECT id, embedding::text FROM {_t(name)} WHERE id IN :ids").bindparams(bindparam("ids", expanding=True))
    with engine().connect() as c:
        rows = c.execute(sql, {"ids": list(ids)}).all()
    return {r[0].strip(): np.asarray([float(x) for x in r[1].strip("[]").split(",")], dtype=np.float32) for r in rows}


def sample_vectors(name: str, limit: int = 1) -> list[list[float]]:
    from sqlalchemy import text
    with engine().connect() as c:
        rows = c.execute(text(f"SELECT embedding::text FROM {_t(name)} LIMIT :n"), {"n": limit}).all()
    return [[float(x) for x in r[0].strip("[]").split(",")] for r in rows]
