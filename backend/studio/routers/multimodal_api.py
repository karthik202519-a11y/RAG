from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import generation, jobs, multimodal
from ..artifacts import get_store
from ..db import get_db
from ..models import User
from ..security import current_user
from .documents import load_document

router = APIRouter(tags=["multimodal"])


def _pdf(db: Session, user: User, document_id: str, role: str = "viewer"):
    d = load_document(db, user, document_id, role)
    ver = d.versions[-1]
    if ver.ext != "pdf":
        raise HTTPException(422, "This feature supports PDF documents only")
    return d, ver


@router.get("/documents/{document_id}/figure-crops")
def figure_crops(document_id: str, page: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Figure regions found on one page (pictures and clusters of drawings), each with its box and a crop image URL."""
    import pymupdf
    d, ver = _pdf(db, user, document_id)
    if page < 1 or (ver.pages and page > ver.pages):
        raise HTTPException(422, f"Page must be within 1-{ver.pages}")
    with pymupdf.open(stream=get_store().get(ver.source_key), filetype="pdf") as doc:
        figs = multimodal.find_figures(doc[page - 1])
    for f in figs:
        f["image_url"] = f"/documents/{d.id}/figure-crops/{page}/{f['n']}.png"
    return {"document_id": d.id, "page": page, "figures": figs}


@router.get("/documents/{document_id}/figure-crops/{page}/{n}.png")
def figure_crop_image(document_id: str, page: int, n: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Response:
    _, ver = _pdf(db, user, document_id)
    try:
        return Response(multimodal.render_crop(get_store().get(ver.source_key), page, n), media_type="image/png")
    except (KeyError, IndexError) as e:
        raise HTTPException(404, str(e)) from None


@router.get("/documents/{document_id}/render/{page}.png")
def render_page(document_id: str, page: int, dpi: int = 100, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Response:
    _, ver = _pdf(db, user, document_id)
    if page < 1 or (ver.pages and page > ver.pages):
        raise HTTPException(404, "No such page")
    return Response(multimodal.render_page(get_store().get(ver.source_key), page, max(40, min(dpi, 200))), media_type="image/png")


class ChartIn(BaseModel):
    page: int = Field(ge=1)
    figure: int | None = Field(None, ge=1, description="Figure number from /figure-crops; default is the whole page")
    model: str | None = Field(None, max_length=100)


@router.post("/documents/{document_id}/chart-to-table")
def chart_to_table(document_id: str, body: ChartIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Ask the vision model to read a chart or table image into a Markdown table. Spends tokens; values are read by a model."""
    _, ver = _pdf(db, user, document_id)
    ok, why = generation.vision_status()
    if not ok:
        raise HTTPException(409, f"Vision model is not configured. {why}")
    data = get_store().get(ver.source_key)
    try:
        png = multimodal.render_crop(data, body.page, body.figure) if body.figure else multimodal.render_page(data, body.page, 150)
    except (KeyError, IndexError) as e:
        raise HTTPException(404, str(e)) from None
    model = body.model or generation.service.settings.azure_vision_deployment
    try:
        res = generation.service.chat_vision(model, multimodal.CHART_PROMPT, png, 900)
    except generation.GenerationError as e:
        raise HTTPException(502, str(e)) from None
    text = res.text.strip()
    return {"page": body.page, "figure": body.figure, "is_chart": not text.upper().startswith("NOT A CHART"), "markdown": text,
            "model": res.model, "tokens": res.prompt_tokens + res.completion_tokens,
            "warning": "Values were read from the image by a model. Check them against the chart before relying on them."}


# ------------------------------------------------------------------------------------ CLIP page search
@router.get("/documents/{document_id}/page-images")
def page_image_status(document_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    from ..vectorstores import faiss_store
    _, ver = _pdf(db, user, document_id)
    ok, why = multimodal.clip_status()
    return {"available": ok, "detail": why, "indexed_pages": faiss_store.count(multimodal.collection(ver.id)), "pages": ver.pages}


@router.post("/documents/{document_id}/page-images", status_code=202)
def build_page_images(document_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Render and embed every page with a local CLIP model so pages can be found by what they look like (background job)."""
    d, ver = _pdf(db, user, document_id, "editor")
    ok, why = multimodal.clip_status()
    if not ok:
        raise HTTPException(409, f"CLIP is not available. {why}")
    job = jobs.enqueue(db, d.project_id, "build_page_images", ver.id)
    return {"job_id": job.id, "pages": ver.pages}


class PageSearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)


@router.post("/documents/{document_id}/page-images/search")
def search_page_images(document_id: str, body: PageSearchIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    from ..vectorstores import faiss_store
    d, ver = _pdf(db, user, document_id)
    if faiss_store.count(multimodal.collection(ver.id)) == 0:
        raise HTTPException(409, "Build the page-image index for this document first")
    hits = multimodal.search_pages(ver.id, body.query, body.top_k)
    for h in hits:
        h["image_url"] = f"/documents/{d.id}/render/{h['page']}.png"
    return {"query": body.query, "results": hits, "note": "Scores are CLIP text-to-image similarity: they rank pages, they are not probabilities."}
