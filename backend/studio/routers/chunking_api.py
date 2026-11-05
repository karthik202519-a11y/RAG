from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..chunking import ChunkingError, run_chunking
from ..chunking.context import use_embedder
from ..embeddings import ProviderError, embed_texts
from ..db import get_db
from ..models import User
from ..security import current_user
from .documents import load_pages

router = APIRouter(tags=["chunking"])


class ChunkConfig(BaseModel):
    method: str
    params: dict = Field(default_factory=dict)
    label: str | None = None


class PreviewIn(BaseModel):
    document_id: str
    version_id: str | None = None
    configs: list[ChunkConfig] = Field(min_length=1, max_length=6)
    page_from: int | None = Field(None, ge=1)
    page_to: int | None = Field(None, ge=1)
    price_per_million_tokens: float | None = Field(None, ge=0, description="Your embedding price, for the cost estimate")


@router.post("/chunking-preview")
def chunking_preview(body: PreviewIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Run one or more chunking configurations on a document and return real chunks with character spans.

    Pass a page range to preview part of a large document. With several configs the same canonical text is
    used for all, so their spans can be compared directly."""
    d, ver, pages = load_pages(db, user, body.document_id, body.version_id)
    lo, hi = body.page_from or 1, body.page_to or len(pages)
    if lo > hi:
        raise HTTPException(422, "page_from must not exceed page_to")
    pages = [p for p in pages if lo <= p["pdf_page"] <= hi]
    if not pages:
        raise HTTPException(422, f"No pages in range {lo}-{hi}")
    results, doc_text = [], None
    for cfg in body.configs:
        try:
            with use_embedder(lambda prov, texts: embed_texts(db, prov, texts).vectors):
                r = run_chunking(pages, cfg.method, cfg.params, body.price_per_million_tokens)
        except ChunkingError as e:
            raise HTTPException(422, f"{cfg.label or cfg.method}: {e}") from None
        except ProviderError as e:
            raise HTTPException(409, f"{cfg.label or cfg.method}: {e}") from None
        doc_text = r.pop("doc_text")
        for c in r["chunks"]:  # the UI slices the shared document text, so don't send it twice
            if c["start"] is not None and c["text"] == doc_text[c["start"]:c["end"]]:
                c["text"] = None
        r["label"] = cfg.label or cfg.method
        results.append(r)
    return {"document_id": d.id, "version_id": ver.id, "filename": d.filename, "page_from": lo, "page_to": hi,
            "doc_text": doc_text, "results": results}
