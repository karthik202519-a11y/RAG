from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User
from ..recommend import recommend, recommend_document
from ..security import current_user, get_project
from .documents import load_document

router = APIRouter(tags=["recommendations"])


@router.get("/projects/{project_id}/recommendations")
def project_recommendations(project_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """A pick for every stage with the reasons and numbers behind it. 'measured' picks come from your evaluation runs; 'rule' picks from
    measurements of your documents; 'default' picks are safe starting points."""
    get_project(db, user, project_id)
    return recommend(db, project_id)


@router.get("/documents/{document_id}/recommendation")
def document_recommendation(document_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    d = load_document(db, user, document_id)
    ver = d.versions[-1]
    if ver.status != "ready":
        return {"profile": None, "chunking": None}
    return recommend_document(db, ver)
