from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..artifacts import get_store
from ..db import get_db
from ..models import User
from ..parsing.compare import COMPARABLE, MAX_PAGES, compare_pdf
from ..security import current_user
from .documents import load_document

router = APIRouter(tags=["parser-lab"])


class CompareIn(BaseModel):
    pages: list[int] = Field(min_length=1, max_length=MAX_PAGES)
    parsers: list[str] = Field(default_factory=lambda: list(COMPARABLE), min_length=1)


@router.post("/documents/{document_id}/parser-compare")
def parser_compare(document_id: str, body: CompareIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Run several parsers on the same PDF pages and report text, table counts, timings and pairwise agreement."""
    d = load_document(db, user, document_id)
    ver = d.versions[-1]
    if ver.ext != "pdf":
        raise HTTPException(422, "Parser comparison supports PDF documents only")
    bad = [p for p in body.parsers if p not in COMPARABLE]
    if bad:
        raise HTTPException(422, f"Unknown parser(s) {bad}; choose from {list(COMPARABLE)}")
    total = ver.pages or 0
    pages = sorted(set(body.pages))
    if total and pages[-1] > total or pages[0] < 1:
        raise HTTPException(422, f"Pages must be within 1-{total}")
    out = compare_pdf(get_store().get(ver.source_key), pages, list(dict.fromkeys(body.parsers)))
    return {"document_id": d.id, "filename": d.filename, **out}
