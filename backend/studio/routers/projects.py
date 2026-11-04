from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..artifacts import get_store
from ..db import get_db
from ..models import AuditEvent, Collection, Document, DocumentVersion, Job, Project, User, Workspace, WorkspaceMember
from ..security import current_user, get_project, workspace_role

router = APIRouter(tags=["projects"])


class ProjectIn(BaseModel):
    workspace_id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = Field("", max_length=2000)


class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field("", max_length=2000)


def project_out(p: Project) -> dict:
    return {"id": p.id, "workspace_id": p.workspace_id, "name": p.name, "description": p.description,
            "created_at": p.created_at.isoformat()}


@router.get("/workspaces")
def list_workspaces(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    rows = db.execute(select(Workspace, WorkspaceMember.role).join(WorkspaceMember)
                      .where(WorkspaceMember.user_id == user.id)).all()
    return [{"id": w.id, "name": w.name, "role": role} for w, role in rows]


@router.post("/projects", status_code=201)
def create_project(body: ProjectIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    role = workspace_role(db, user.id, body.workspace_id)
    if role is None:
        raise HTTPException(404, "Workspace not found")
    if role == "viewer":
        raise HTTPException(403, "Requires editor role")
    p = Project(workspace_id=body.workspace_id, name=body.name.strip(), description=body.description)
    db.add(p)
    db.add(AuditEvent(user_id=user.id, workspace_id=p.workspace_id, action="project.create", target=p.id))
    db.commit()
    return project_out(p)


@router.get("/projects")
def list_projects(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    rows = db.scalars(select(Project).join(WorkspaceMember, WorkspaceMember.workspace_id == Project.workspace_id)
                      .where(WorkspaceMember.user_id == user.id).order_by(Project.created_at.desc())).all()
    return [project_out(p) for p in rows]


@router.get("/projects/{project_id}")
def read_project(project_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    p = get_project(db, user, project_id)
    docs = db.scalar(select(func.count(Document.id)).where(Document.project_id == p.id)) or 0
    return {**project_out(p), "documents": docs,
            "collections": [{"id": c.id, "name": c.name, "description": c.description} for c in p.collections]}


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    p = get_project(db, user, project_id, "owner")
    get_store().delete_prefix(f"projects/{p.id}")
    db.query(Job).filter(Job.project_id == p.id).delete()
    db.query(DocumentVersion).filter(DocumentVersion.project_id == p.id).delete()
    db.add(AuditEvent(user_id=user.id, workspace_id=p.workspace_id, action="project.delete", target=p.id))
    db.delete(p)
    db.commit()


@router.post("/projects/{project_id}/collections", status_code=201)
def create_collection(project_id: str, body: CollectionIn, user: User = Depends(current_user),
                      db: Session = Depends(get_db)) -> dict:
    p = get_project(db, user, project_id, "editor")
    c = Collection(project_id=p.id, name=body.name.strip(), description=body.description)
    db.add(c)
    db.commit()
    return {"id": c.id, "name": c.name, "description": c.description}


@router.get("/projects/{project_id}/collections")
def list_collections(project_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    p = get_project(db, user, project_id)
    return [{"id": c.id, "name": c.name, "description": c.description} for c in p.collections]
