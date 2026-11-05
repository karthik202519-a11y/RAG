"""Context expansion after retrieval: return the larger passage (parent) or the neighbouring chunks of each hit."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import IndexChunk, IndexParent

KINDS = ("none", "parent", "neighbors")


def format_citation(filename: str, pages: list[int], printed: list) -> str:
    span = "" if not pages else (f"p.{pages[0]}" if len(pages) == 1 else f"pp.{pages[0]}-{pages[-1]}")
    printed = [p for p in (printed or []) if p]
    out = filename + (f", PDF {span}" if span else "")
    if printed:
        out += f" (printed {printed[0]}" + (f"-{printed[-1]}" if len(printed) > 1 else "") + ")"
    return out


def expand_results(db: Session, hits: list[dict], kind: str, neighbors: int = 1, limit: int | None = None) -> list[dict]:
    """parent: replace each hit's text by its parent passage; hits sharing a parent collapse into the best-ranked one.
    neighbors: wrap each hit in the `neighbors` chunks on either side. The matched chunk's own text is kept as `child_text`."""
    if kind == "none" or not hits:
        return hits[:limit] if limit else hits
    if kind == "parent":
        ids = {h.get("parent_id") for h in hits if h.get("parent_id")}
        parents = {p.id: p for p in db.scalars(select(IndexParent).where(IndexParent.id.in_(ids or [""])))}
        out, seen = [], {}
        for h in hits:
            par = parents.get(h.get("parent_id"))
            if par is None:
                out.append(dict(h, expanded="none"))
                continue
            if par.id in seen:
                seen[par.id]["matched_children"] += 1
                continue
            e = dict(h, expanded="parent", child_text=h["text"], text=par.text, tokens=par.tokens, pages=par.pages,
                     matched_children=1, start=None, end=None)
            e["citation"] = format_citation(par.filename, par.pages, h.get("printed_pages"))
            seen[par.id] = e
            out.append(e)
        for i, h in enumerate(out, 1):
            h["rank"] = i
        return out[:limit] if limit else out
    if kind == "neighbors":
        out = []
        for h in hits[:limit] if limit else hits:
            lo, hi = h["chunk_index"] - neighbors, h["chunk_index"] + neighbors
            sib = db.scalars(select(IndexChunk).where(IndexChunk.index_id == h["index_id"], IndexChunk.version_id == h["version_id"],
                                                      IndexChunk.chunk_index >= lo, IndexChunk.chunk_index <= hi)
                             .order_by(IndexChunk.chunk_index)).all()
            if len(sib) <= 1:
                out.append(dict(h, expanded="none"))
                continue
            pages = sorted({p for s in sib for p in (s.pages or [])})
            e = dict(h, expanded="neighbors", child_text=h["text"], text="\n\n".join(s.text for s in sib), pages=pages,
                     tokens=sum(s.tokens for s in sib), neighbor_chunks=[s.chunk_index for s in sib], start=None, end=None)
            e["citation"] = format_citation(h["filename"], pages, h.get("printed_pages"))
            out.append(e)
        return out
    raise ValueError(f"expand must be one of {KINDS}")
