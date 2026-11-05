# Architecture

```
 Browser (Next.js 16, React 19, TanStack Query)
        │  REST + Server-Sent Events
        ▼
 FastAPI app (backend/studio)         ┌────────────────────────────┐
  routers ─ auth / projects /         │ Method registry            │
            documents / jobs /  ────▶ │ (machine-readable specs,   │──▶ UI selectors and forms
            catalog / chunking        │  status: runs / not built) │    are generated from this
        │                             └────────────────────────────┘
        ├─ SQLAlchemy ─▶ SQLite (local default) | PostgreSQL (Docker)      application metadata
        ├─ Artifact store ─▶ local folder | S3-compatible (MinIO)          originals + parsed JSON
        └─ Job runner ─▶ thread pool (local) | Celery on Redis (Docker)    parse jobs
```

## Principles

- **Local first, scale by configuration.** With no environment variables the studio runs on SQLite, a local
  artifact folder and an in-process job pool. `DATABASE_URL`, `REDIS_URL`, `S3_ENDPOINT` switch each part
  to the Docker profile; the same test-suite passes on both databases.
- **The registry is the contract.** A method is advertised through `registry/*_specs.py`. Statuses are
  `implemented`, `requires_credentials` or `unavailable` (which must carry a reason). A test asserts that every
  advertised chunker exists in code, and the UI renders unavailable methods as unusable. Nothing pretends.
- **Authorization lives in queries.** Every resource resolves through `get_project(user, id, min_role)`;
  unknown and forbidden are indistinguishable (404).
- **Jobs are durable and idempotent.** State is in the `jobs` table. A finished job is never re-run, a failed
  one retries with exponential backoff and then becomes `dead`, a crash re-queues work at startup, and a
  cancelled job returns its subject to an honest state.
- **Artifacts are immutable per version.** `projects/<project>/versions/<version>/{source.<ext>, extraction.json,
  visual_pages/*.png}`. New bytes make a new version; nothing is overwritten.

## Data flow (implemented)

1. `POST /projects/{id}/documents` → validate type, size, magic bytes → SHA-256 → duplicate check →
   store original → create `Document` / `DocumentVersion` → enqueue `ingest_document`.
2. Worker: read original → parser by extension (`parsing/`) → pages of text blocks and Markdown tables →
   store `extraction.json` → record pages, characters, tokens, tables, parser, seconds → `ready`.
3. `POST /chunking-preview` → load pages (optionally a page range) → `chunking.build_canon` joins items into one
   canonical text with offsets → a method returns character spans → the engine measures tokens, pages,
   overlap, mid-sentence endings, duplication, coverage → response.
4. The UI slices the shared canonical text with those spans to draw exact colored chunk boundaries.

## Retrieval, generation and evaluation

```
question ─▶ query transform (LLM, optional) ─▶ retrieve (dense | BM25 | hybrid, per query) ─▶ fuse (RRF)
         ─▶ MMR (optional) ─▶ LLM rerank (optional) ─▶ top-k with per-stage contributions
                                  │
                                  ├─▶ grounded answer: numbered passages in a token budget -> chat model -> citation check
                                  └─▶ evaluation: labelled questions -> metrics -> paired comparison between runs
```

- `retrieval/service.py` is one retriever; `retrieval/pipeline.py` composes stages and reports each one with timing and
  LLM tokens, so a query transformation is never confused with the search it feeds.
- `jobs/index_job.py` builds an index: chunk the frozen document versions, embed through the cache, write to Qdrant.
  A retry starts clean, so a half-built index can never be searched.
- `jobs/figures_job.py` writes vision descriptions back into the extraction, checkpointed per page.
- `evaluation/metrics.py` holds the pure metric functions; `evaluation/service.py` imports questions, runs them and
  compares runs; `jobs/eval_job.py` executes a run in the background.
- Labels are (document fragment or id, pages), not chunk ids, so one labelled dataset can score any index.

## Deployment

`docker compose up -d` starts Postgres, Redis, Qdrant and MinIO. `--profile app` adds the API, a Celery worker (same image)
and the web UI. With `DATABASE_URL`, `REDIS_URL`, `QDRANT_URL` and `S3_ENDPOINT` set the app uses all four; with none set it
runs on SQLite, an embedded Qdrant, a local folder and a thread pool.

## Layout

```
backend/studio/
  config.py db.py models.py security.py artifacts.py main.py
  registry/    method specs (chunking_specs.py, parser_specs.py)
  parsing/     pdf.py (PyMuPDF), textlike.py (txt, md, docx, html, csv, json)
  chunking/    canon.py tokens.py methods.py engine.py structure.py
  embeddings/  providers.py (Azure OpenAI, offline demo) service.py (cache)
  vectorstores/  qdrant_store.py
  retrieval/   service.py (dense, BM25, hybrid) pipeline.py (transforms, MMR, rerank)
  generation/  service.py (grounded answers, vision)
  evaluation/  metrics.py service.py
  jobs/        runner.py tasks.py index_job.py figures_job.py eval_job.py celery_app.py
  routers/     auth, projects, documents, jobs_api, catalog, chunking_api, indexes, explorer, evaluation
backend/alembic/  migrations         backend/tests/  76 tests
frontend/src/    app/ (pages)  components/  lib/ (api, auth, project, planned)
app/ scripts/    earlier prototype: embedding + dense retrieval + evaluation (not yet merged into studio/)
```

## Adding a method

1. Implement it (for chunkers: a function in `chunking/methods.py` taking `(canon, params)` and returning cuts).
2. Register it in `registry/chunking_specs.py` with parameters, ranges, advantages, disadvantages, limitations.
3. Add it to `METHODS`. The registry test fails if the two disagree.
4. The UI picks it up automatically, including its parameter form and explanation panel.
