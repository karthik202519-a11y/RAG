from collections import defaultdict, deque
from time import monotonic

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import AuditEvent, User, Workspace, WorkspaceMember
from ..security import create_token, current_user, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

_FAILS: dict[str, deque] = defaultdict(deque)
MAX_FAILS, WINDOW = 5, 60.0


def _throttled(key: str) -> bool:
    q = _FAILS[key]
    while q and monotonic() - q[0] > WINDOW:
        q.popleft()
    return len(q) >= MAX_FAILS


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    display_name: str = Field(min_length=1, max_length=200)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


def _user_out(u: User) -> dict:
    return {"id": u.id, "email": u.email, "display_name": u.display_name}


@router.post("/register", status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)) -> dict:
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "An account with this email already exists")
    user = User(email=email, display_name=body.display_name, password_hash=hash_password(body.password))
    ws = Workspace(name=f"{body.display_name}'s workspace")
    db.add_all([user, ws])
    db.flush()
    db.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="owner"))
    db.add(AuditEvent(user_id=user.id, workspace_id=ws.id, action="user.register", target=user.id))
    db.commit()
    return {"token": create_token(user.id), "user": _user_out(user)}


@router.post("/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)) -> dict:
    email = body.email.lower()
    key = f"{email}|{request.client.host if request.client else ''}"
    if _throttled(key):
        raise HTTPException(429, "Too many failed attempts; try again in a minute")
    user = db.scalar(select(User).where(User.email == email))
    if not user or not user.is_active or not verify_password(body.password, user.password_hash):
        _FAILS[key].append(monotonic())
        raise HTTPException(401, "Incorrect email or password")
    _FAILS.pop(key, None)
    return {"token": create_token(user.id), "user": _user_out(user)}


@router.get("/me")
def me(user: User = Depends(current_user)) -> dict:
    return _user_out(user)
