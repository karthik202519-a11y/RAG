from fastapi import APIRouter, HTTPException

from ..registry import registry

router = APIRouter(prefix="/method-catalog", tags=["method-catalog"])


@router.get("")
def list_methods(category: str | None = None, status: str | None = None, q: str | None = None) -> dict:
    items = registry.list(category)
    if status:
        items = [m for m in items if m.status == status]
    if q:
        ql = q.lower()
        items = [m for m in items if ql in m.name.lower() or ql in m.summary.lower() or ql in m.id]
    return {"items": [m.model_dump() for m in items], "categories": registry.categories(),
            "counts": {s: sum(1 for m in registry.list(category) if m.status == s)
                       for s in ("implemented", "requires_credentials", "unavailable")}}


@router.get("/{method_id}")
def read_method(method_id: str) -> dict:
    m = registry.get(method_id)
    if m is None:
        raise HTTPException(404, "Unknown method")
    return m.model_dump()
