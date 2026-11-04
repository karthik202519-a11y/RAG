"""HTTP API for the RAG pipeline: upload -> extract -> chunk -> embed -> search.

Run:  uvicorn app.main:app --reload     then open http://127.0.0.1:8000/docs
"""
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__, config, embeddings, retrieval
from .chunking import build_chunks
from .extraction import extract_pdf
from .store import NotFound, Store

app = FastAPI(
    title="RAG-I",
    version=__version__,
    description="Upload PDFs, chunk them, embed the chunks and retrieve cited passages.",
)
store = Store()

SizeQuery = Query(config.DEFAULT_CHUNK_SIZE, ge=config.MIN_CHUNK_SIZE, le=config.MAX_CHUNK_SIZE,
                  description="Target chunk size in tokens")


class ChunkRequest(BaseModel):
    size: int = Field(config.DEFAULT_CHUNK_SIZE, ge=config.MIN_CHUNK_SIZE, le=config.MAX_CHUNK_SIZE)
    overlap: int | None = Field(None, ge=0, description="Overlap in tokens; default size // 8")


class EmbedRequest(BaseModel):
    size: int = Field(config.DEFAULT_CHUNK_SIZE, ge=config.MIN_CHUNK_SIZE, le=config.MAX_CHUNK_SIZE)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)
    chunk_size: int = Field(config.DEFAULT_CHUNK_SIZE, ge=config.MIN_CHUNK_SIZE, le=config.MAX_CHUNK_SIZE)
    doc_ids: list[str] | None = Field(None, description="Restrict to these documents; default all")
    types: list[Literal["text", "table"]] | None = Field(None, description="Restrict to chunk types")


@app.exception_handler(NotFound)
async def _not_found(_, exc: NotFound):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=404, content={"detail": f"Document '{exc.args[0]}' not found"})


@app.exception_handler(embeddings.EmbeddingError)
async def _embedding_error(_, exc: embeddings.EmbeddingError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ------------------------------------------------------------------- health
@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "version": __version__, "embedding_backend": config.EMBEDDING_BACKEND,
            "documents": len(store.list())}


# ---------------------------------------------------------------- documents
@app.post("/documents", status_code=201, tags=["documents"])
def upload_document(file: UploadFile = File(...), extract: bool = True) -> dict:
    """Upload a PDF. Same file bytes -> same doc_id (re-uploading is harmless).

    With extract=true (default) the PDF is parsed immediately; a 300-page PDF takes
    a minute or two. Use extract=false and call POST /documents/{id}/extract later.
    """
    content = file.file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(415, "Only PDF files are supported")
    meta, created = store.create(file.filename or "document.pdf", content)
    if extract and not meta["extracted"]:
        meta = _extract(meta["doc_id"])
    meta["created"] = created
    return meta


def _extract(doc_id: str) -> dict:
    meta = store.get(doc_id)
    try:
        pages = extract_pdf(store.source_path(doc_id), meta["filename"], store.doc_dir(doc_id) / "visual_pages")
    except Exception as e:  # corrupt / encrypted PDF
        raise HTTPException(422, f"Could not read the PDF: {type(e).__name__}: {e}") from e
    store.save_pages(doc_id, pages)
    meta.update(
        extracted=True,
        pages=len(pages),
        tables=sum(1 for p in pages for i in p["items"] if i["type"] == "table"),
        text_blocks=sum(1 for p in pages for i in p["items"] if i["type"] == "text"),
        needs_vision_pages=[p["pdf_page"] for p in pages if p["needs_vision"]],
        chunked={}, embedded={},  # anything derived from old pages is stale
    )
    store.save_meta(doc_id, meta)
    return meta


@app.post("/documents/{doc_id}/extract", tags=["documents"])
def extract_document(doc_id: str) -> dict:
    """Parse the PDF into pages of text blocks and tables (re-runs reset chunks and vectors)."""
    store.get(doc_id)
    return _extract(doc_id)


@app.get("/documents", tags=["documents"])
def list_documents() -> list[dict]:
    return store.list()


@app.get("/documents/{doc_id}", tags=["documents"])
def get_document(doc_id: str) -> dict:
    return store.get(doc_id)


@app.delete("/documents/{doc_id}", status_code=204, tags=["documents"])
def delete_document(doc_id: str) -> None:
    store.delete(doc_id)


@app.get("/documents/{doc_id}/pages/{page}", tags=["documents"])
def get_page(doc_id: str, page: int) -> dict:
    """The extracted content of one PDF page (1-based): what the chunker sees."""
    try:
        pages = store.load_pages(doc_id)
    except FileNotFoundError:
        raise HTTPException(409, "Document not extracted yet") from None
    if not 1 <= page <= len(pages):
        raise HTTPException(404, f"Page {page} out of range 1..{len(pages)}")
    return pages[page - 1]


@app.get("/documents/{doc_id}/pages/{page}/image", tags=["documents"])
def get_page_image(doc_id: str, page: int):
    """PNG render of a diagram/chart page (only pages flagged needs_vision have one)."""
    f = store.doc_dir(doc_id) / "visual_pages" / f"page_{page}.png"
    if not f.exists():
        raise HTTPException(404, "No image for this page (only needs_vision pages are rendered)")
    return FileResponse(f, media_type="image/png")


# ------------------------------------------------------------------ chunking
@app.post("/documents/{doc_id}/chunk", tags=["pipeline"])
def chunk_document(doc_id: str, req: ChunkRequest = ChunkRequest()) -> dict:
    """Split the extracted pages into chunks of about `size` tokens. Re-chunking a size
    replaces its chunks and invalidates that size's vectors."""
    meta = store.get(doc_id)
    if not meta["extracted"]:
        raise HTTPException(409, "Document not extracted yet; call POST /documents/{id}/extract")
    chunks = build_chunks(store.load_pages(doc_id), req.size, req.overlap)
    for c in chunks:
        c["doc_id"] = doc_id
    store.save_chunks(doc_id, req.size, chunks)
    tokens = sorted(c["tokens"] for c in chunks)
    stats = {"chunks": len(chunks),
             "text": sum(c["type"] == "text" for c in chunks),
             "table": sum(c["type"] == "table" for c in chunks),
             "tokens_median": tokens[len(tokens) // 2] if tokens else 0,
             "tokens_max": tokens[-1] if tokens else 0}
    meta["chunked"][str(req.size)] = stats
    meta["embedded"].pop(str(req.size), None)
    store.save_meta(doc_id, meta)
    return {"doc_id": doc_id, "size": req.size, **stats}


@app.get("/documents/{doc_id}/chunks", tags=["pipeline"])
def list_chunks(doc_id: str, size: int = SizeQuery, offset: int = Query(0, ge=0),
                limit: int = Query(20, ge=1, le=200), type: Literal["text", "table"] | None = None) -> dict:
    try:
        chunks = store.load_chunks(doc_id, size)
    except FileNotFoundError as e:
        raise HTTPException(409, f"{e}; call POST /documents/{{id}}/chunk first") from None
    if type:
        chunks = [c for c in chunks if c["type"] == type]
    return {"doc_id": doc_id, "size": size, "total": len(chunks), "offset": offset,
            "chunks": chunks[offset:offset + limit]}


# ----------------------------------------------------------------- embedding
@app.post("/documents/{doc_id}/embed", tags=["pipeline"])
def embed_document(doc_id: str, req: EmbedRequest = EmbedRequest()) -> dict:
    """Embed every chunk of the given size. Already-embedded texts come from the cache (not billed)."""
    meta = store.get(doc_id)
    try:
        chunks = store.load_chunks(doc_id, req.size)
    except FileNotFoundError:
        raise HTTPException(409, f"Not chunked at size {req.size}; call POST /documents/{{id}}/chunk first") from None
    matrix, tokens = embeddings.embed_texts([embeddings.embed_text(c) for c in chunks])
    store.save_vectors(doc_id, req.size, matrix)
    info = {"chunks": len(chunks), "dim": int(matrix.shape[1]), "backend": config.EMBEDDING_BACKEND,
            "tokens_billed": tokens}
    meta["embedded"][str(req.size)] = info
    store.save_meta(doc_id, meta)
    return {"doc_id": doc_id, "size": req.size, **info}


@app.post("/documents/{doc_id}/process", tags=["pipeline"])
def process_document(doc_id: str, req: EmbedRequest = EmbedRequest()) -> dict:
    """Convenience: extract (if needed) -> chunk -> embed in one call."""
    meta = store.get(doc_id)
    if not meta["extracted"]:
        _extract(doc_id)
    chunked = chunk_document(doc_id, ChunkRequest(size=req.size))
    embedded = embed_document(doc_id, req)
    return {"document": store.get(doc_id), "chunk": chunked, "embed": embedded}


# ----------------------------------------------------------------- retrieval
@app.post("/search", tags=["retrieval"])
def search(req: SearchRequest) -> dict:
    """Return the top-k chunks for a query, each with score, pages and a citation string."""
    try:
        return retrieval.search(store, req.query, req.top_k, req.chunk_size, req.doc_ids, req.types)
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
