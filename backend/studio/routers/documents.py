import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from .. import generation, jobs
from ..artifacts import get_store
from ..chunking.canon import build_canon
from ..config import settings
from ..db import get_db
from ..models import AuditEvent, Document, DocumentVersion, Job, User
from ..parsing import parser_for, parser_options, supported_extensions
from ..parsing.textlike import CODE_EXTS
from ..security import current_user, get_project

router = APIRouter(tags=["documents"])


def _magic_ok(ext: str, head: bytes, whole: bytes) -> bool:
    if ext == "pdf":
        return head.startswith(b"%PDF-")
    if ext == "png":
        return head.startswith(b"\x89PNG")
    if ext in ("jpg", "jpeg"):
        return head.startswith(b"\xff\xd8")
    if ext in ("docx", "pptx", "xlsx"):
        part = {"docx": "word/document.xml", "pptx": "ppt/presentation.xml", "xlsx": "xl/workbook.xml"}[ext]
        try:
            return zipfile.is_zipfile(BytesIO(whole)) and part in zipfile.ZipFile(BytesIO(whole)).namelist()
        except zipfile.BadZipFile:
            return False
    if ext in ("txt", "md", "markdown", "csv", "json", "html", "htm", "xml") or ext in CODE_EXTS:
        return b"\x00" not in head  # text formats must not contain NUL bytes
    return False


def version_out(v: DocumentVersion) -> dict:
    return {"id": v.id, "version": v.version, "sha256": v.sha256, "size_bytes": v.size_bytes, "ext": v.ext,
            "status": v.status, "error": v.error, "pages": v.pages, "char_count": v.char_count,
            "token_count": v.token_count, "table_count": v.table_count, "parser": v.parser,
            "extraction_seconds": v.extraction_seconds, "created_at": v.created_at.isoformat()}


def document_out(d: Document, detail: bool = False) -> dict:
    latest = d.versions[-1] if d.versions else None
    out = {"id": d.id, "project_id": d.project_id, "collection_id": d.collection_id, "filename": d.filename,
           "tags": d.tags, "doc_metadata": d.doc_metadata, "created_at": d.created_at.isoformat(),
           "latest_version": version_out(latest) if latest else None, "version_count": len(d.versions)}
    if detail:
        out["versions"] = [version_out(v) for v in d.versions]
    return out


def load_document(db: Session, user: User, document_id: str, min_role: str = "viewer") -> Document:
    d = db.get(Document, document_id)
    if d is None:
        raise HTTPException(404, "Document not found")
    get_project(db, user, d.project_id, min_role)  # 404 if the user has no access to the project
    return d


@router.get("/supported-formats")
def supported_formats() -> dict:
    return {"extensions": supported_extensions(), "max_upload_mb": settings.max_upload_mb}


@router.post("/projects/{project_id}/documents", status_code=201)
async def upload_documents(project_id: str, files: list[UploadFile] = File(...),
                           collection_id: str | None = Form(None), tags: str = Form(""), parser: str = Form(""),
                           user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Upload one or many files. Each file is hashed; identical content in this project is reported as a
    duplicate and not stored again. A same-named file with new content becomes a new version.
    Parsing runs as a background job."""
    project = get_project(db, user, project_id, "editor")
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    store, results = get_store(), []
    limit = settings.max_upload_mb * 1024 * 1024
    for f in files:
        name = Path(f.filename or "document").name
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if parser_for(ext) is None:
            results.append({"filename": name, "status": "rejected",
                            "error": f"Unsupported file type '.{ext}'. Supported: {', '.join(supported_extensions())}"})
            continue
        if parser and parser != parser_for(ext) and parser not in parser_options(ext):
            results.append({"filename": name, "status": "rejected",
                            "error": f"Parser '{parser}' cannot read .{ext} files. Choose one of: {', '.join([parser_for(ext), *parser_options(ext)])}"})
            continue
        data = await f.read(limit + 1)
        if len(data) > limit:
            results.append({"filename": name, "status": "rejected", "error": f"Larger than {settings.max_upload_mb} MB"})
            continue
        if not data:
            results.append({"filename": name, "status": "rejected", "error": "Empty file"})
            continue
        if not _magic_ok(ext, data[:1024], data):
            results.append({"filename": name, "status": "rejected",
                            "error": f"File content does not look like a .{ext} file"})
            continue
        digest = hashlib.sha256(data).hexdigest()
        dup = db.scalar(select(DocumentVersion).where(DocumentVersion.project_id == project.id,
                                                      DocumentVersion.sha256 == digest))
        if dup is not None:
            results.append({"filename": name, "status": "duplicate", "document_id": dup.document_id,
                            "version_id": dup.id})
            continue
        doc = db.scalar(select(Document).where(Document.project_id == project.id, Document.filename == name))
        if doc is None:
            doc = Document(project_id=project.id, collection_id=collection_id, filename=name, tags=tag_list)
            db.add(doc)
            db.flush()
        n = (db.scalar(select(func.max(DocumentVersion.version)).where(DocumentVersion.document_id == doc.id)) or 0) + 1
        ver = DocumentVersion(document_id=doc.id, project_id=project.id, version=n, sha256=digest,
                              size_bytes=len(data), ext=ext, source_key="pending")
        db.add(ver)
        db.flush()
        ver.source_key = f"projects/{project.id}/versions/{ver.id}/source.{ext}"
        store.put(ver.source_key, data)
        db.add(AuditEvent(user_id=user.id, workspace_id=project.workspace_id, action="document.upload",
                          target=ver.id, detail={"filename": name, "bytes": len(data)}))
        db.commit()
        job = jobs.enqueue(db, project.id, "ingest_document", ver.id, {"parser": parser} if parser else None)
        results.append({"filename": name, "status": "queued", "document_id": doc.id, "version_id": ver.id,
                        "version": n, "job_id": job.id})
    return {"results": results}


@router.get("/projects/{project_id}/documents")
def list_documents(project_id: str, q: str | None = None, status: str | None = None, ext: str | None = None,
                   collection_id: str | None = None, tag: str | None = None,
                   limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                   user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    get_project(db, user, project_id)
    stmt = select(Document).where(Document.project_id == project_id).options(selectinload(Document.versions))
    if q:
        stmt = stmt.where(Document.filename.ilike(f"%{q}%"))
    if collection_id:
        stmt = stmt.where(Document.collection_id == collection_id)
    if status or ext:
        sub = select(DocumentVersion.document_id)
        if status:
            sub = sub.where(DocumentVersion.status == status)
        if ext:
            sub = sub.where(DocumentVersion.ext == ext.lower())
        stmt = stmt.where(Document.id.in_(sub))
    docs = db.scalars(stmt.order_by(Document.created_at.desc())).all()
    if tag:
        docs = [d for d in docs if tag in (d.tags or [])]
    total = len(docs)
    return {"total": total, "items": [document_out(d) for d in docs[offset:offset + limit]]}


@router.get("/documents/{document_id}")
def read_document(document_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    return document_out(load_document(db, user, document_id), detail=True)


class DocumentPatch(BaseModel):
    filename: str | None = Field(None, min_length=1, max_length=500)
    tags: list[str] | None = None
    doc_metadata: dict | None = None
    collection_id: str | None = None


@router.patch("/documents/{document_id}")
def patch_document(document_id: str, body: DocumentPatch, user: User = Depends(current_user),
                   db: Session = Depends(get_db)) -> dict:
    d = load_document(db, user, document_id, "editor")
    if body.filename is not None:
        d.filename = Path(body.filename).name
    if body.tags is not None:
        d.tags = [t.strip() for t in body.tags if t.strip()]
    if body.doc_metadata is not None:
        d.doc_metadata = body.doc_metadata
    if body.collection_id is not None:
        d.collection_id = body.collection_id or None
    db.commit()
    return document_out(d, detail=True)


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    d = load_document(db, user, document_id, "editor")
    store = get_store()
    for v in d.versions:
        store.delete_prefix(f"projects/{d.project_id}/versions/{v.id}")
        db.query(Job).filter(Job.subject_id == v.id).delete()
    db.add(AuditEvent(user_id=user.id, action="document.delete", target=d.id, detail={"filename": d.filename}))
    db.delete(d)
    db.commit()


def load_pages(db: Session, user: User, document_id: str, version_id: str | None = None):
    d = load_document(db, user, document_id)
    ver = next((v for v in d.versions if v.id == version_id), None) if version_id else d.versions[-1]
    if ver is None:
        raise HTTPException(404, "Version not found")
    if ver.status != "ready" or not ver.extraction_key:
        raise HTTPException(409, f"Document is {ver.status}; extraction is not ready")
    return d, ver, json.loads(get_store().get(ver.extraction_key))


@router.get("/documents/{document_id}/content")
def document_content(document_id: str, version_id: str | None = None, page: int | None = Query(None, ge=1),
                     user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Extracted content: the canonical text plus per-page items (what the chunkers see)."""
    d, ver, pages = load_pages(db, user, document_id, version_id)
    if page is not None:
        if page > len(pages):
            raise HTTPException(404, f"Page {page} out of range 1..{len(pages)}")
        return {"document_id": d.id, "version_id": ver.id, "page": pages[page - 1]}
    canon = build_canon(pages)
    return {"document_id": d.id, "version_id": ver.id, "filename": d.filename, "pages": len(pages),
            "text": canon.text, "needs_vision_pages": [p["pdf_page"] for p in pages if p["needs_vision"]]}


@router.get("/documents/{document_id}/pages/{page}/image")
def page_image(document_id: str, page: int, version_id: str | None = None, user: User = Depends(current_user),
               db: Session = Depends(get_db)) -> Response:
    d = load_document(db, user, document_id)
    ver = next((v for v in d.versions if v.id == version_id), None) if version_id else d.versions[-1]
    key = f"projects/{d.project_id}/versions/{ver.id}/visual_pages/page_{page}.png" if ver else None
    if not key or not get_store().exists(key):
        raise HTTPException(404, "No rendered image for this page (only diagram/chart pages are rendered)")
    return Response(get_store().get(key), media_type="image/png")


class DescribeIn(BaseModel):
    pages: list[int] | None = Field(None, description="Only these PDF pages; default every flagged page")
    model: str | None = Field(None, max_length=100, description="Vision deployment; default AZURE_OPENAI_VISION_DEPLOYMENT")
    force: bool = Field(False, description="Describe again even if a description already exists (costs tokens)")


@router.get("/documents/{document_id}/figures")
def list_figures(document_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Diagram and chart pages flagged at ingestion, and which have a vision description."""
    from ..jobs.figures_job import figure_status
    d, ver, pages = load_pages(db, user, document_id)
    figs = figure_status(pages)
    pending = db.scalar(select(func.count(Job.id)).where(Job.subject_id == ver.id, Job.kind == "describe_figures",
                                                         Job.status.in_(("queued", "running")))) or 0
    return {"document_id": d.id, "version_id": ver.id, "figures": figs, "described": sum(f["described"] for f in figs),
            "pending": pending > 0,
            "vision": {"configured": generation.vision_status()[0], "detail": generation.vision_status()[1]}}


@router.post("/documents/{document_id}/describe-figures", status_code=202)
def describe_figures(document_id: str, body: DescribeIn = DescribeIn(), user: User = Depends(current_user),
                     db: Session = Depends(get_db)) -> dict:
    """Describe diagram/chart pages with a vision model (background job; costs tokens). Descriptions become searchable text."""
    from ..jobs.figures_job import figure_status
    d = load_document(db, user, document_id, "editor")
    ok, why = generation.vision_status()
    if not ok:
        raise HTTPException(409, f"Vision model is not configured. {why}")
    _, ver, pages = load_pages(db, user, document_id)
    flagged = [f["page"] for f in figure_status(pages)]
    if not flagged:
        raise HTTPException(422, "No diagram or chart pages were flagged in this document.")
    bad = sorted(set(body.pages or []) - set(flagged))
    if bad:
        raise HTTPException(422, f"Pages {bad} are not flagged as figure pages. Flagged: {flagged}")
    job = jobs.enqueue(db, d.project_id, "describe_figures", ver.id,
                       {"pages": body.pages, "model": body.model, "force": body.force})
    db.add(AuditEvent(user_id=user.id, action="document.describe_figures", target=d.id, detail={"pages": body.pages or "all"}))
    db.commit()
    return {"job_id": job.id, "pages": body.pages or flagged, "model": body.model or generation.service.settings.azure_vision_deployment}


@router.get("/documents/{document_id}/source")
def download_source(document_id: str, version_id: str | None = None, user: User = Depends(current_user),
                    db: Session = Depends(get_db)) -> Response:
    d = load_document(db, user, document_id)
    ver = next((v for v in d.versions if v.id == version_id), None) if version_id else d.versions[-1]
    if ver is None:
        raise HTTPException(404, "Version not found")
    media = {"pdf": "application/pdf", "html": "text/html", "json": "application/json"}.get(ver.ext, "application/octet-stream")
    return Response(get_store().get(ver.source_key), media_type=media,
                    headers={"Content-Disposition": f'inline; filename="{d.filename}"',
                             "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox"})


@router.get("/projects/{project_id}/ingestion-stats")
def ingestion_stats(project_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    get_project(db, user, project_id)
    vs = db.scalars(select(DocumentVersion).where(DocumentVersion.project_id == project_id)).all()
    by_ext: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for v in vs:
        by_ext[v.ext] = by_ext.get(v.ext, 0) + 1
        by_status[v.status] = by_status.get(v.status, 0) + 1
    done = [v for v in vs if v.status == "ready"]
    finished = len(done) + by_status.get("failed", 0)
    secs = sum(v.extraction_seconds or 0 for v in done)
    jobs_by = dict(db.execute(select(Job.status, func.count()).where(Job.project_id == project_id)
                              .group_by(Job.status)).all())
    return {
        "documents": db.scalar(select(func.count(Document.id)).where(Document.project_id == project_id)) or 0,
        "versions": len(vs), "pages": sum(v.pages or 0 for v in done),
        "characters": sum(v.char_count or 0 for v in done), "tokens": sum(v.token_count or 0 for v in done),
        "tables": sum(v.table_count or 0 for v in done), "storage_bytes": sum(v.size_bytes for v in vs),
        "by_type": by_ext, "by_status": by_status,
        "extraction_success_rate": round(len(done) / finished, 3) if finished else None,
        "throughput_pages_per_second": round(sum(v.pages or 0 for v in done) / secs, 2) if secs else None,
        "queue_depth": jobs_by.get("queued", 0) + jobs_by.get("running", 0), "jobs": jobs_by,
        "estimated_ingestion_cost_usd": 0.0,  # local parsers have no per-document cost
    }
