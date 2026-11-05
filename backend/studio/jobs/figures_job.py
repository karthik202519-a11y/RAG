"""Job handler: describe diagram/chart pages with a vision model and write the descriptions into the extraction.

Pages flagged `needs_vision` at ingestion have a PNG stored. Each description is added to that page as a text item
beginning "[Figure description, page N]" so every chunker and the index builder can use it. Extraction is saved
after every page (checkpoint): a cancel, a crash or a retry never repeats paid work, and never loses it.
"""
import json

from .. import generation
from ..artifacts import get_store
from ..chunking.canon import build_canon
from ..chunking.tokens import count
from ..config import settings
from ..models import DocumentVersion
from .runner import JobContext, PermanentJobError, handler

PROMPT = (
    "You are preparing a figure from a technical document for a search index. Describe it so that someone could answer "
    "questions about it without seeing it. Include: the title or caption; every label, box, axis title, unit and legend "
    "entry, written exactly as shown; the relationships between elements (for example 'Requirements -> Design'); the kind "
    "of figure (flowchart, bar chart, scatter plot, map, table, schematic); and for charts the key values, ranges and "
    "trends you can actually read. Do not guess values you cannot read; say they are unreadable. Plain text, no preamble."
)
MARK = "[Figure description, page"


def figure_status(pages: list[dict]) -> list[dict]:
    out = []
    for p in pages:
        if not p.get("needs_vision"):
            continue
        desc = next((i for i in p["items"] if i.get("source") == "vision"), None)
        out.append({"page": p["pdf_page"], "has_image": bool(p.get("image_path")), "described": desc is not None,
                    "model": desc.get("model") if desc else None, "description": desc["text"] if desc else None})
    return out


@handler("describe_figures")
def describe_figures(ctx: JobContext) -> None:
    db, job = ctx.db, ctx.job
    ver = db.get(DocumentVersion, job.subject_id)
    if ver is None or not ver.extraction_key:
        raise PermanentJobError("Document version is not extracted")
    ok, why = generation.vision_status()
    if not ok:
        raise PermanentJobError(f"Vision model is not configured: {why}")
    model = job.params.get("model") or settings.azure_vision_deployment
    force = bool(job.params.get("force"))
    wanted = set(job.params.get("pages") or [])
    store = get_store()
    pages = json.loads(store.get(ver.extraction_key))
    todo = [p for p in pages if p.get("needs_vision") and (not wanted or p["pdf_page"] in wanted)
            and (force or not any(i.get("source") == "vision" for i in p["items"]))]
    prefix = f"projects/{ver.project_id}/versions/{ver.id}"
    done = skipped = tin = tout = 0
    notes: list[str] = []
    ctx.progress(0.02, f"{len(todo)} page(s) to describe")
    for n, page in enumerate(todo):
        if ctx.cancelled():
            raise InterruptedError("cancelled")
        num = page["pdf_page"]
        key = f"{prefix}/visual_pages/page_{num}.png"
        if not store.exists(key):
            skipped += 1
            notes.append(f"page {num}: no rendered image")
            continue
        res = generation.service.chat_vision(model, PROMPT, store.get(key))
        text = res.text.strip()
        page["items"] = [i for i in page["items"] if i.get("source") != "vision"]
        if text:
            page["items"].append({"type": "text", "text": f"{MARK} {num}] {text}", "source": "vision", "model": res.model})
            done += 1
        else:
            skipped += 1
            notes.append(f"page {num}: the model returned no text")
        tin, tout = tin + res.prompt_tokens, tout + res.completion_tokens
        store.put(ver.extraction_key, json.dumps(pages, ensure_ascii=False).encode("utf-8"))  # checkpoint
        canon = build_canon(pages)
        ver.char_count, ver.token_count = len(canon.text), count(canon.text)
        db.commit()
        ctx.progress(0.05 + 0.93 * (n + 1) / len(todo), f"described page {num} ({n + 1}/{len(todo)})")
    job.params = {**job.params, "result": {"described": done, "skipped": skipped, "model": model,
                                           "prompt_tokens": tin, "completion_tokens": tout, "notes": notes}}
    db.commit()
