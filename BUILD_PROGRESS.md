# Build progress

Honest log of what is done, what is verified, and what is outstanding. Updated as work lands.
Product spec: the RAG Learning Studio master prompt. Scope is very large (17 screens, ~100 methods,
12 database adapters, GraphRAG, evaluation, load testing), so it is being built in vertical slices.

## Slice 1: runnable foundation + ingestion + chunking lab

### Done and verified

| Item | How it was verified |
|---|---|
| Backend: FastAPI, SQLAlchemy 2, Alembic, auth, workspaces/projects/collections | 76 automated tests; the same 76 pass on SQLite and on PostgreSQL 16 in Docker |
| Method registry (24 chunking + 19 parser entries) with education metadata, validation and status honesty | tests assert registry == implementations and every unavailable entry explains why |
| 9 chunking methods with exact character spans, statistics, validation | unit tests + run on the real 297-page NASA handbook (all 9 produce full-coverage chunks) |
| Document ingestion API (7 file types, hashing, versions, validation) | tests incl. corrupt PDF, wrong magic bytes, path tricks, 40 files per request |
| Job system: retries with exponential backoff, dead state, manual retry, cancel, idempotency, restart recovery | tests, including a simulated crash |
| Celery + Redis execution | Manual: API and worker as separate processes, jobs crossed Redis, results visible in the API |
| Docker Compose infrastructure | Postgres, Redis, Qdrant, MinIO started healthy |
| Frontend: Next.js 16 + Tailwind 4 + TanStack Query; shell, auth, projects, ingestion, chunking playground, method catalog, "not built yet" screens | `tsc` and `eslint` clean; driven in a real browser (register, create project, upload, live progress, cancel, retry, chunk, compare, statistics) |
| Light/dark themes, keyboard-reachable controls, labelled inputs | manual |

### Bugs found by testing and fixed

1. One-page PDF returned no text: header/footer detection removed every line (every line is on 100% of pages).
2. Test and app used different database engines after reconfiguration; migrations ran on the wrong DB.
3. Cancelled parse left the document showing "processing" forever; now returns to "uploaded" with a note (a stand-in handler hid this in the first test, which is why the hook now applies to any non-ready version).
4. UI: picked files vanished (the live `FileList` was cleared before React read it). Found only by driving the browser.
5. UI: histogram labels overlapped; status badge truncated.
6. PyMuPDF 1.28.2 finds 359 "tables" in the NASA handbook where PyMuPDF 2.0 + layout finds 40; versions are pinned.

### Known limitations (not yet addressed)

- **PDF parsing is slow with PyMuPDF 2.0**: roughly 1.3 s/page on the NASA handbook; the job reports progress.
- The 10,000-document load test has **not** been run.
- S3/MinIO artifact store: code exists, never run against the live MinIO container.
- Application Dockerfiles do not exist; `docker compose --profile app` will fail until they do.
- MinIO's official images are gone from Docker Hub and quay.io; compose uses a community mirror (`bitnamilegacy/minio`).
- Structure-aware chunks are located in the text by string search; unlocated chunks are listed but not highlighted (the UI warns).
- Access tokens are stored in `localStorage` and passed as `?token=` for the SSE stream. Fine for local use; harden before exposing publicly.
- No rate limiting beyond login throttling; no file scanning hook; no audit viewer.

## Slice 2: embeddings, Qdrant indexes, retrieval lab

### Done and verified

- Index builds as background jobs (chunk, embed with cache, write to Qdrant) with progress, retry, cancel, dead state.
- 92 backend tests pass (SQLite; the Slice 1 subset also passes on PostgreSQL).
- Live on the real NASA handbook with Azure embeddings. On the 23 labelled questions (rank 1 / top 5): dense 17 / 20, BM25 8 / 17, hybrid 13 / 20.
- Browser-verified: Indexes page, Retrieval Lab search, source highlighting, compare-modes.

### Bugs found by testing and fixed

1. BM25 dropped true matches in tiny corpora (inverse document frequency is exactly 0 for a term in half the chunks); a match is now "contains a query term".
2. Continued-table chunks could not be located in the document, so they had no pages and were cited as just "nasa.pdf". Tables are now located by their whole Markdown block, and unlocated chunks fall back to the chunker's own pages. 150 sampled results: 0 without a page citation.
3. A test raced between "version ready" and "job succeeded".

### Findings worth knowing

- Reciprocal-rank-fusion hybrid did **not** improve on dense search for these natural-language questions. That is a measured result on one document set, not a general claim.
- Structure-aware chunks that merely contain added text (captions, notes) used to be invisible to the highlighter; locating by verbatim lines fixed it.

### Answer generation (added in this slice)

- Grounded prompt with numbered passages inside a token budget; dropped passages are reported; the exact messages are shown in the UI.
- Citation markers are checked against the passages actually sent; refusals and a score gate that avoids the model call are supported.
- 110 tests pass (13 for generation, using a fake chat model). Live: `gpt-4o-mini` answered the PDR question with a resolving citation and refused an off-topic one.
- Reasoning models (`gpt-5` family) accept no temperature and spend output tokens thinking; the code handles the first and the UI warns about the second.

### Outstanding from this slice

- Qdrant server mode is untested by the app; other vector stores and embedding providers are not built.
- Answers are not streamed or saved; faithfulness is not scored.
- Retrieval quality is measured by an ad-hoc script, not yet by the Evaluation Center.
- The PDR answer was correct but thin ("defined in Table G-6"): the cited table chunk is large, so the model had little to quote. Row-level table chunks are the likely fix; not yet tried.

## Next slices (in order)

2. ~~Baseline retrieval and grounded answers~~ (done above). **Next: the Evaluation Center** (question sets, recall/MRR/nDCG, faithfulness), then reranking and row-level table chunks.
3. Embedding explorer, retrieval inspector, more chunkers (semantic, parent-child, contextual).
4. BM25 + hybrid + reranking; pgvector/Chroma adapters.
5. Parser lab, tables, multimodal, GraphRAG.
6. Evaluation datasets and metrics, experiment studio, comparison dashboard.
7. Scale and hardening: load test, application containers, security review.
