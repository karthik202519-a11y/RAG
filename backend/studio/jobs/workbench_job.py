"""Execute typed recipes without blocking the worker pool waiting on nested jobs."""
import time
from types import SimpleNamespace
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified
from ..models import WorkbenchRun, Index, EvalRun, GraphBuild, now
from ..embeddings import get_provider
from ..retrieval import pipeline
from ..generation import service as gen
from .runner import handler, JobContext, DEAD_HOOKS, CANCEL_HOOKS, PermanentJobError
from . import index_job, eval_job, graph_job


class InlineContext:
    def __init__(self, parent, subject_id, params=None):
        self.db, self.parent = parent.db, parent
        self.job = SimpleNamespace(subject_id=subject_id, params=params or {})
    def progress(self, fraction, stage=None):
        self.parent.progress(min(.9, .1 + fraction*.7), stage)
    def cancelled(self):
        return self.parent.cancelled()


def stage(ctx, row, name, execute, detail):
    if ctx.cancelled():
        raise InterruptedError("cancelled")
    artifact = {"stage": name, "status": "running", "detail": detail}
    position = len(row.artifacts or [])
    row.artifacts = [*(row.artifacts or []), artifact]
    ctx.db.commit()
    ctx.progress(.05, name)
    start = time.perf_counter()
    try:
        result = execute()
    except Exception:
        artifact = {**artifact, "status": "failed", "ms": round((time.perf_counter()-start)*1000, 1)}
        row.artifacts = [*row.artifacts[:position], artifact]
        ctx.db.commit()
        raise
    artifact = {**artifact, "status": "done", "ms": round((time.perf_counter()-start)*1000, 1)}
    row.artifacts = [*row.artifacts[:position], artifact]
    ctx.db.commit()
    return result


@handler("workbench")
def run_workbench(ctx: JobContext):
    db = ctx.db
    row = db.get(WorkbenchRun, ctx.job.subject_id)
    if row is None:
        raise PermanentJobError("Run no longer exists")
    if row.status == "done":
        return
    row.status, row.error = "running", None
    row.result = row.result or {"cases": []}
    db.commit()
    try:
        for n, recipe in enumerate(row.config["recipes"]):
            cfg = recipe["config"]
            frozen = cfg["index"]
            cases = list(row.result.get("cases", []))
            case = cases[n] if n < len(cases) else {"pipeline_id": recipe["id"], "name": recipe["name"], "version": recipe["version"]}
            idx = db.get(Index, case.get("index_id")) if case.get("index_id") else None
            if idx is None and row.kind == "pipeline" and cfg.get("reuse_index_id"):
                idx = db.get(Index, cfg["reuse_index_id"])
                if idx is None or idx.status != "ready":
                    raise PermanentJobError("The reused index is no longer ready. Save a new pipeline version.")
            if idx is None:
                provider = get_provider(frozen["provider"])
                if provider.model != frozen["model"]:
                    raise PermanentJobError("The embedding model changed after the recipe was saved. Save a new recipe.")
                idx = Index(project_id=row.project_id, name=f"{row.name}: {recipe['name']} v{recipe['version']}"[:200], provider=frozen["provider"],
                            model=frozen["model"], store=frozen["store"], collection="pending", config=frozen)
                db.add(idx); db.flush()
                idx.collection = f"idx_{idx.id}"
            case["index_id"] = idx.id
            cases = [*cases[:n], case, *cases[n+1:]]
            row.result = {"cases": cases}
            db.commit()
            stage(ctx, row, "index", lambda: index_job.build_index(InlineContext(ctx, idx.id)),
                  {"index_id": idx.id, "recipe": recipe["name"], "chunker": frozen["chunking"]["method"], "provider": idx.provider, "store": idx.store, "versions": frozen["version_ids"], "reuse": bool(cfg.get("reuse_index_id")) and row.kind == "pipeline"})
            case["build_seconds"] = idx.build_seconds
            case["embedding_tokens_billed"] = idx.tokens_billed
            case["embedding_cache_hits"] = idx.cache_hits
            retrieval = cfg["retrieval"]
            if retrieval["mode"] in ("graph", "graph_hybrid"):
                graph = db.get(GraphBuild, idx.id)
                if graph is None:
                    graph = GraphBuild(index_id=idx.id, model=cfg["answer_model"], status="queued")
                    db.add(graph); db.commit()
                if graph.status != "ready":
                    stage(ctx, row, "graph extraction", lambda: graph_job.build_graph(InlineContext(ctx, idx.id, {"max_chunks": 50, "min_tokens": 1})),
                          {"model": graph.model, "max_chunks": 50, "note": "Real model extraction; one call per sampled chunk."})
            if row.kind == "experiment":
                er = db.get(EvalRun, case.get("eval_run_id")) if case.get("eval_run_id") else None
                if er is None:
                    er = EvalRun(dataset_id=row.config["dataset_id"], index_id=idx.id, index_name=idx.name, label=f"{recipe['name']} v{recipe['version']}",
                        config={"mode": retrieval["mode"], "k": retrieval["top_k"], "split": row.config["split"], "transform": retrieval["transform"],
                                "rerank": retrieval["rerank"], "mmr": retrieval["mmr"], "expand": retrieval["expand"], "neighbors": retrieval["neighbors"],
                                "llm_model": cfg["answer_model"], "with_answers": row.config["with_answers"], "judge": False, **{k: retrieval[k] for k in ("document_ids", "types", "score_threshold", "rrf_k", "n_queries", "mmr_lambda", "rerank_candidates")}})
                    db.add(er); db.flush()
                    case["eval_run_id"] = er.id
                    row.result = {"cases": [*cases[:n], case, *cases[n+1:]]}
                    db.commit()
                stage(ctx, row, "evaluate", lambda: eval_job.run_eval(InlineContext(ctx, er.id)), {"eval_run_id": er.id, "questions": row.config["questions"], "dataset_snapshot": row.config["dataset_id"]})
                db.refresh(er)
                if er.status != "done" or er.metrics.get("failed_questions", 0):
                    raise PermanentJobError("Evaluation has failed questions; inspect the saved evaluation run.")
                case.update(metrics=er.metrics, seconds=er.seconds, per_question=er.per_question)
            else:
                kwargs = {k: retrieval[k] for k in ("mode", "top_k", "transform", "n_queries", "mmr_lambda", "rerank", "rerank_candidates", "llm_model", "document_ids", "types", "score_threshold", "rrf_k", "expand", "neighbors")}
                kwargs["mmr_on"] = retrieval["mmr"]
                result = stage(ctx, row, "retrieve", lambda: pipeline.run(db, idx, retrieval["query"], **kwargs), {"query": retrieval["query"], **kwargs})
                case["retrieval"] = result
                if cfg["with_answer"]:
                    case["answer"] = stage(ctx, row, "answer", lambda: gen.answer(db, idx, retrieval["query"], model=cfg["answer_model"],
                        mode=retrieval["mode"], top_k=min(20, retrieval["top_k"]), retrieved=result), {"model": cfg["answer_model"], "context": "The exact transformed, reranked and expanded retrieval result above."})
            row.result = {"cases": [*row.result["cases"][:n], case, *row.result["cases"][n+1:]]}
            flag_modified(row, "result")
            db.commit()
        row.status = "done"
        db.commit()
        ctx.progress(1, "done")
    except Exception as e:
        db.rollback()
        row = db.get(WorkbenchRun, ctx.job.subject_id)
        row.status, row.error = "failed", f"{type(e).__name__}: {str(e)[:500]}"
        if "idx" in locals() and idx is not None:
            failed_index = db.get(Index, idx.id)
            if failed_index and failed_index.status == "building":
                failed_index.status, failed_index.error = "failed", row.error
            failed_graph = db.get(GraphBuild, idx.id)
            if failed_graph and failed_graph.status in ("queued", "running"):
                failed_graph.status, failed_graph.error = "failed", row.error
        db.commit()
        if isinstance(e, InterruptedError):
            raise
        raise PermanentJobError(row.error) from e


def failed(db, job, error):
    row = db.get(WorkbenchRun, job.subject_id)
    if row:
        row.status, row.error = "failed", error[:500]
        db.commit()


DEAD_HOOKS["workbench"] = failed
def cancelled(db, job):
    row = db.get(WorkbenchRun, job.subject_id)
    if row:
        row.status, row.error = "cancelled", "Execution cancelled. Completed artifacts are retained."
        db.commit()


CANCEL_HOOKS["workbench"] = cancelled
