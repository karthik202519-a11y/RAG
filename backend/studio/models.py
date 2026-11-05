"""Persistent entities for the foundation slice.

Every project-owned row carries its project (and through it the workspace), so
authorization is enforced in queries, not only in the UI.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def uid() -> str:
    return uuid.uuid4().hex


def now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(200))
    password_hash: Mapped[str] = mapped_column(String(300))
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Workspace(Base):
    __tablename__ = "workspaces"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"
    __table_args__ = (UniqueConstraint("workspace_id", "user_id"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # owner | editor | viewer


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    collections: Mapped[list["Collection"]] = relationship(cascade="all, delete-orphan")


class Collection(Base):
    __tablename__ = "collections"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Document(Base):
    """A logical document. Its content lives in DocumentVersion rows (new bytes = new version)."""
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    collection_id: Mapped[str | None] = mapped_column(ForeignKey("collections.id", ondelete="SET NULL"), index=True)
    filename: Mapped[str] = mapped_column(String(500))
    tags: Mapped[list] = mapped_column(JSON, default=list)
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    versions: Mapped[list["DocumentVersion"]] = relationship(
        cascade="all, delete-orphan", order_by="DocumentVersion.version")


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "version"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str] = mapped_column(String(32), index=True)  # denormalised for fast duplicate checks
    version: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    ext: Mapped[str] = mapped_column(String(16))
    source_key: Mapped[str] = mapped_column(String(500))  # artifact-store key of the original bytes
    status: Mapped[str] = mapped_column(String(16), default="uploaded")  # uploaded|processing|ready|failed
    error: Mapped[str | None] = mapped_column(Text)
    pages: Mapped[int | None] = mapped_column(Integer)
    char_count: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)
    table_count: Mapped[int | None] = mapped_column(Integer)
    parser: Mapped[str | None] = mapped_column(String(64))
    extraction_key: Mapped[str | None] = mapped_column(String(500))  # artifact-store key of parsed JSON
    extraction_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(64))  # ingest_document
    subject_id: Mapped[str] = mapped_column(String(32), index=True)  # e.g. document_version id
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    # queued | running | succeeded | failed | cancelled | dead
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EmbeddingCache(Base):
    """One embedding per (provider, model, text). Billed once, reused across indexes and re-runs."""
    __tablename__ = "embedding_cache"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256(provider|model|text)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(200))
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # float32, little endian
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Index(Base):
    """An immutable index version: which chunks (method + params + documents), embedded by which model,
    stored in which vector collection. Changing any of these means a new Index, never an edit."""
    __tablename__ = "indexes"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    config: Mapped[dict] = mapped_column(JSON)  # frozen: chunking, embedding, version_ids, store
    status: Mapped[str] = mapped_column(String(16), default="building")  # building|ready|failed
    error: Mapped[str | None] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(200))
    dimension: Mapped[int | None] = mapped_column(Integer)
    metric: Mapped[str] = mapped_column(String(16), default="cosine")
    store: Mapped[str] = mapped_column(String(32), default="qdrant")
    collection: Mapped[str] = mapped_column(String(100))
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    tokens_embedded: Mapped[int | None] = mapped_column(Integer)
    tokens_billed: Mapped[int | None] = mapped_column(Integer)  # new provider tokens (cache hits cost nothing)
    cache_hits: Mapped[int | None] = mapped_column(Integer)
    build_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class IndexChunk(Base):
    __tablename__ = "index_chunks"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)  # also the Qdrant point id
    index_id: Mapped[str] = mapped_column(ForeignKey("indexes.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str] = mapped_column(String(32), index=True)
    version_id: Mapped[str] = mapped_column(String(32), index=True)
    filename: Mapped[str] = mapped_column(String(500))
    chunk_index: Mapped[int] = mapped_column(Integer)
    start: Mapped[int | None] = mapped_column(Integer)
    end: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    ctype: Mapped[str] = mapped_column(String(16))
    tokens: Mapped[int] = mapped_column(Integer)
    pages: Mapped[list] = mapped_column(JSON)
    printed_pages: Mapped[list] = mapped_column(JSON)
    section: Mapped[str | None] = mapped_column(Text)
    parent_id: Mapped[str | None] = mapped_column(String(32), index=True)  # set by the parent-child chunker
    context: Mapped[str | None] = mapped_column(Text)  # model-written situating sentence, embedded with the chunk (contextual indexes)


class IndexParent(Base):
    """The larger passage a small chunk belongs to (small-to-big retrieval). Not embedded; returned for context."""
    __tablename__ = "index_parents"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    index_id: Mapped[str] = mapped_column(ForeignKey("indexes.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str] = mapped_column(String(32))
    version_id: Mapped[str] = mapped_column(String(32))
    filename: Mapped[str] = mapped_column(String(500))
    text: Mapped[str] = mapped_column(Text)
    tokens: Mapped[int] = mapped_column(Integer)
    pages: Mapped[list] = mapped_column(JSON)
    section: Mapped[str | None] = mapped_column(Text)


class RetrievalRun(Base):
    """Every search is recorded so runs can be compared later."""
    __tablename__ = "retrieval_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    index_id: Mapped[str] = mapped_column(ForeignKey("indexes.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(String(32))
    query: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(16))
    params: Mapped[dict] = mapped_column(JSON)
    result_ids: Mapped[list] = mapped_column(JSON)
    timings_ms: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class EvalDataset(Base):
    __tablename__ = "eval_datasets"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class EvalQuestion(Base):
    """A labelled question. `evidence` lists where the answer lives as {document, pages}; `document` is a
    document id or a filename fragment, so labels survive re-uploads and work across indexes."""
    __tablename__ = "eval_questions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("eval_datasets.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    question: Mapped[str] = mapped_column(Text)
    qtype: Mapped[str] = mapped_column(String(40), default="")
    answerable: Mapped[bool] = mapped_column(default=True)
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    reference_answer: Mapped[str | None] = mapped_column(Text)
    split: Mapped[str] = mapped_column(String(12), default="test")  # train | validation | test
    approved: Mapped[bool] = mapped_column(default=True)  # generated questions need human approval


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("eval_datasets.id", ondelete="CASCADE"), index=True)
    index_id: Mapped[str] = mapped_column(String(32), index=True)
    index_name: Mapped[str] = mapped_column(String(200))
    label: Mapped[str] = mapped_column(String(200), default="")
    config: Mapped[dict] = mapped_column(JSON)  # frozen: mode, k, split, rerank ...
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    error: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    per_question: Mapped[list | None] = mapped_column(JSON)
    seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class GraphBuild(Base):
    """State of the knowledge graph extracted from one index's chunks."""
    __tablename__ = "graph_builds"
    index_id: Mapped[str] = mapped_column(ForeignKey("indexes.id", ondelete="CASCADE"), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|ready|failed
    model: Mapped[str] = mapped_column(String(100), default="")
    chunks_requested: Mapped[int] = mapped_column(Integer, default=0)
    chunks_processed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_failed: Mapped[int] = mapped_column(Integer, default=0)
    relations_dropped: Mapped[int] = mapped_column(Integer, default=0)  # relations whose entities were not in the chunk text
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class GraphEntity(Base):
    __tablename__ = "graph_entities"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    index_id: Mapped[str] = mapped_column(ForeignKey("indexes.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(300))
    norm: Mapped[str] = mapped_column(String(300), index=True)
    etype: Mapped[str] = mapped_column(String(60), default="")
    mentions: Mapped[int] = mapped_column(Integer, default=1)  # chunks the entity was found in


class GraphEdge(Base):
    __tablename__ = "graph_edges"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    index_id: Mapped[str] = mapped_column(ForeignKey("indexes.id", ondelete="CASCADE"), index=True)
    src_id: Mapped[str] = mapped_column(ForeignKey("graph_entities.id", ondelete="CASCADE"), index=True)
    dst_id: Mapped[str] = mapped_column(ForeignKey("graph_entities.id", ondelete="CASCADE"), index=True)
    relation: Mapped[str] = mapped_column(String(200))
    chunk_id: Mapped[str] = mapped_column(String(32), index=True)  # provenance: the chunk the relation was read from


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    user_id: Mapped[str | None] = mapped_column(String(32), index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(100))
    target: Mapped[str | None] = mapped_column(String(200))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class PipelineRecipe(Base):
    __tablename__ = "pipeline_recipes"
    __table_args__ = (UniqueConstraint("project_id", "name", "version"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    version: Mapped[int] = mapped_column(Integer)
    config: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class WorkbenchRun(Base):
    __tablename__ = "workbench_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    name: Mapped[str] = mapped_column(String(160))
    config: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    artifacts: Mapped[list] = mapped_column(JSON, default=list)
    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
