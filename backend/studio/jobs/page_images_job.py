"""Job handler: embed every page of a PDF with a local CLIP model."""
from .. import multimodal
from ..artifacts import get_store
from ..models import DocumentVersion
from .runner import JobContext, PermanentJobError, handler


@handler("build_page_images")
def build_page_images(ctx: JobContext) -> None:
    ver = ctx.db.get(DocumentVersion, ctx.job.subject_id)
    if ver is None or ver.ext != "pdf":
        raise PermanentJobError("The document is gone or is not a PDF")
    ok, why = multimodal.clip_status()
    if not ok:
        raise PermanentJobError(why)
    data = get_store().get(ver.source_key)
    multimodal.build_page_index(ver.id, data, progress=lambda f, s: ctx.progress(f, s), cancelled=ctx.cancelled)
