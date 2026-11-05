# Build progress

## Latest UI refinement

- Fluid working-page widths, collapsible labelled navigation/icon rail and compact workflow links.
- Expandable lessons/method/construction evidence keep primary controls visible.
- Storage selector plus separate sample/column/relationship views; row and native schema details expand on request.
- Shorter pipeline canvas, latest saved experiment comparison, compact index summaries, chunking method selector and separate Settings views.
- Fixed mobile ingestion filters and an absolutely positioned accessible table label that escaped its scroll container.
- Frontend lint, TypeScript and production build passed; browser checked desktop, tablet and actual390CSS mobile layouts and focused working flows. Backend behavior was unchanged for this refinement.


## Latest: guided redesign, workbench and Storage Explorer

- Implemented both formerly planned screens and removed the empty planned navigation section; old URLs redirect to the working routes.
- Added immutable pipeline recipes and durable executions in forward migration `0007`. Applied to the verified local SQLite app database; new tables preserve existing projects and indexes.
- Background execution builds or reuses real indexes, records stage artifacts and generates answers from exact retrieval results. Experiments copy approved questions and use an isolated index per saved recipe.
- Added TF-IDF and unified graph/graph-hybrid retrieval, plus 15 discoverable search recipes. The selector exposes previously scattered query transforms, expansion, MMR and reranking.
- Redesigned shared surfaces/navigation and added guided explanations, method guides, source-backed graph construction and actual embedding input/vector samples.
- Added a small synthetic project, two saved comparison recipes, a four-question benchmark and real local-model execution. A live five-chunk graph build yielded 29 entities and 22 relations; these are model proposals tied to source text, not independently verified facts.
- The first full run found a terminal-state job race and a brittle wall-clock assertion. The race is repaired; the concurrency test now checks matching outputs, overlapping requests and the configured concurrency bound. The final suite passed: **504 tests, zero failures and zero skips** (`output/backend-redesign-tests.xml`). PostgreSQL focused workbench and regression checks also passed.

- Added read-only Storage Explorer: 23 application tables, scoped real samples, primary/foreign keys, logical links and reasons for each table. Native records are read from all five engines, with actual vector previews and source-chunk links. Tiny FastEmbed examples are ready in the synthetic project for every store.
- Storage checks: seven passed on SQLite and PostgreSQL (`output/storage-tests.xml`, `output/postgres-storage-tests.xml`); browser verified all five native stores and record navigation. Desktop and actual 390 CSS pixel screens reviewed: ship, no open material findings.
- Renamed the product to **RAG** in UI, browser metadata, API title and project documentation. API restarted and browser confirmed HTTP 200.

## Earlier verification ? 2026-11-05

- Complete backend run: **476 passed, zero failures, zero skips**, including real CLIP inference. Evidence: `output/backend-tests.xml`.
- PostgreSQL application database: **170 matrix cases passed**. SQLite: the same **170 cases passed** as part of the full suite. Each matrix covers 23 chunkers ? five real stores (115), seven embedding-provider ? store wiring cases (35), and four chat-provider ? store cases (20).
- Additional real FastEmbed + cross-encoder tests: **five passed**, one for each store. Evidence: `output/local-model-tests.xml`.
- **Live Azure smoke: all five stores passed** ingestion, indexing, dense/BM25/hybrid retrieval and a real cited answer. The document is synthetic and tiny; this is functional evidence, not a quality benchmark. Evidence: `output/live-store-smoke.json`.
- App runtime: all five stores are available. Added a persistent dedicated pgvector service on localhost:5435, configured the local app URL, and verified write/search/delete through Settings. The existing PostgreSQL container was left intact.
- Hosted providers other than Azure remain simulated-response verification only; keys/servers are absent. Full matrix model-dependent tasks are explicitly simulated. No claim of all possible parameter combinations or production scale is made.
- Frontend lint, type checking and production build passed. Four primary screens fit a 390px viewport; all 14 main routes loaded without page exceptions or error alerts. Final browser interactions confirmed both directions of the mobile focus trap, Escape focus restoration, all 23 chunkers and five stores in selectors, keyword search, real local reranking with neighbor expansion, and the Chroma connection check. Evidence: `output/browser-verification.json`. See the audit section below for usability changes and remaining backlog.

The sections below retain historical results; their old counts and limitations are superseded by this section and the current feature matrix where applicable.


Honest log of what is done, what is verified, and what is outstanding. Updated as work lands.
Product spec: the RAG master prompt. Scope is very large (17 screens, ~100 methods,
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

## Slice 3: everything else that could be built and verified, then evaluation

Built, with automated tests (167 pass on SQLite and on PostgreSQL 16) and live checks on the real NASA handbook:
row-level table chunking, query transforms (rewrite, multi-query, HyDE, decomposition), MMR, LLM reranking, vision
descriptions of diagram pages, PPTX/XLSX/XML parsers, the Embedding Explorer, the Evaluation Center, and containers for
the whole application.

### Measured results (the same 34 labelled NASA questions; 31 answerable; hit/MRR at k=5)

| Configuration | Hit | MRR | nDCG | vs dense (paired MRR, 95% CI) |
|---|---|---|---|---|
| Dense | 0.871 | 0.743 | 0.763 | baseline |
| BM25 | 0.710 | 0.495 | 0.535 | -0.248 [-0.44, -0.07]: **real** |
| Hybrid (RRF) | 0.903 | 0.703 | 0.741 | -0.040 [-0.16, +0.06]: noise |
| Dense + LLM rerank | 0.839 | 0.716 | 0.735 | -0.027 [-0.10, +0.05]: noise |
| Dense + HyDE | 0.871 | 0.688 | 0.734 | -0.055 [-0.20, +0.08]: noise |
| Dense on the older index without figure text | 0.871 | 0.748 | 0.767 | +0.005 [0.00, +0.02]: noise |

Reading it honestly: on these questions plain dense search is a strong baseline, BM25 alone is clearly worse, and
hybrid, reranking and HyDE are not distinguishable from it. This is one document set and 31 questions, with the test
split not held out, so it supports "BM25 alone is worse here" and nothing stronger. Figure descriptions did not move the
page-hit metrics (the diagram pages were already retrieved) but changed an answer: a flowchart question that was refused
(`N7`) was answered correctly once the page description was indexed. Page-hit metrics cannot see that kind of gain.

### Bugs found by testing and fixed

1. **nDCG above 1** (and average precision above 1): several chunks from one labelled page each earned credit while the
   ideal ranking had one per evidence item. Found by the live evaluation, fixed by crediting each evidence item once, with
   a regression test and a randomized bounds test.
2. **Dependencies missing from `requirements.txt`** (Qdrant, BM25, NumPy, OpenAI, PowerPoint, Excel): the containers
   crashed on start. Fixed, with a test that checks every import is declared.
3. HyDE and rewrite were *added to* the original query, so a weak original diluted the better one; they now replace it
   (HyDE keeps the original only for keyword matching).
4. The frontend image could not build offline because fonts were fetched from Google at build time; switched to the
   self-hosted Geist package.
5. Dashboard copy still said embeddings and evaluation were "not in the studio".
6. Continued-table chunks, BM25 on tiny corpora, cancelled-parse state and an import-order cycle (Slices 1 and 2).

### Known limits

- The 10,000-document load test has not been run.
- Faithfulness comes from a model judge (claims checked against passages, quotes verified in code); it is an opinion, not ground truth. Answer correctness is not measured.
- GraphRAG extraction keeps only entities found verbatim in the chunk, so it rejects most model proposals (445 rejected vs 139 relations kept on the handbook sample) and leaves many entities unconnected. Its value for answers has not been evaluated.
- Contextual embedding and the FAISS store are built but not yet compared with the plain index on the evaluation set.
- pdfplumber and pypdf are comparison-only; ingestion uses PyMuPDF, which found no tables on handbook pages where pdfplumber found six (worth investigating).
- Structure-aware chunking still produces large table chunks; the row-level method exists but has not been compared in
  the Evaluation Center on a table-heavy question set.
- Figure descriptions are page-level, unverified model output, and cost tokens.
- Frontend image is 1.9 GB; API image is 850 MB.
- Only `gpt-4o-mini` and `gpt-4o` were exercised end to end; `gpt-5` models need a larger output-token limit.

## Latest slice: every method, suggestions, scale test

- All 23 chunking methods run (11 added), the parser choice on upload (pdfplumber, local OCR, Azure Document Intelligence), 5 vector stores, 8 embedding providers, a local cross-encoder, 5 chat providers, and the Visual Lab (figure crops, chart-to-table, CLIP page search).
- Suggestions page and per-document suggestion chip: a pick per stage, with reasons and numbers, labelled Measured / Rule / Starting point.
- Scale test on real files (docs/SCALE_TEST.md): chunking on up to 8.4M characters, end-to-end at 10,000 characters. **Not run at 1M+ end to end**, by request.
- Bugs found by the work: chunk highlights covered only a fragment of blocks with internal line breaks (JSON, Markdown); fixed with a regression test. The 1M chunking run reported 2 failed documents for structure-aware and table-rows that did not recur and whose cause is unknown.
- The registry had stale "planned" entries for things that already existed (MMR, HyDE, parent-document, vision ...); all corrected. 16 of 91 methods remain unavailable, each with its reason.

### Known limits (added)

- Hosted-service adapters (OpenAI, Anthropic, Gemini, Cohere, Voyage, Ollama, Azure Document Intelligence) are tested against recorded responses, not live services.
- Semantic chunking with the local embedder ran at 5 to 11 sentences per second on this CPU, so it is impractical at millions of characters without a GPU or a hosted embedder.
- The Suggestions page's "Measured" label needs 10+ evaluation questions and a gap outside the confidence interval; with fewer it falls back to rules and says so.

## Next slices (in order)

1. Run the evaluation grid on the handbook to compare plain, contextual, row-level-table, semantic and parent-child indexes, and FAISS against Qdrant; add UI controls for parent/neighbour expansion.
2. Investigate why PyMuPDF missed tables on handbook pages where pdfplumber found them (the parser option now exists on upload).
3. Run the end-to-end scale test at 1M, 3M and 10M characters, Milvus/Weaviate/Elasticsearch stores, Docling, global GraphRAG search, visual pipeline builder.
4. The 10,000-document load test, OpenTelemetry export, rate limits and a security review.


## End-to-end compatibility and usability audit ? 2026-11-05

Current audit scope: the implemented catalog, with the historical 10,000-character scale limit preserved.
This is a synthetic integration matrix, not an evaluation of all handbook questions or a production load test.

- Added `backend/tests/test_end_to_end_matrix.py`: all 23 chunkers ? five real vector stores. Each combination exercises ingestion, index creation, stored-vector inspection, dense/BM25/hybrid retrieval, parent/neighbor expansion, filters, all five query transforms ? three rerank settings with MMR, cited answers, evaluation, and deletion. Native HTML/table, JSON, code and PDF fixtures are selected for structured methods.
- Added seven embedding-provider ? five store wiring checks, and four real chat adapters ? five stores with simulated API transport. Model-dependent matrix stages are simulated and labelled in the report; they do not establish live hosted-service quality.
- Fixed non-Azure default-model discovery, availability in the UI, contextual/proposition model checks, contextual indexing, query transforms, judge defaults and graph model selection. Fixed Docker Compose forwarding of optional provider configuration.
- Query/document embeddings and their caches are separated for Cohere/Voyage. Querying an index with a different configured embedding model is rejected even if vector dimensions match.
- LanceDB saves its metric in table metadata; older non-cosine indexes must be rebuilt. Cross-encoder malformed-score failures retain original results and warn.
- Settings has live checks for all five vector stores and individual embedding providers. Unavailable stores are rejected before enqueueing an index job.
- UI: six-stage learning navigation, clear dashboard next action, progressive disclosure for index/advanced retrieval options, context expansion controls, model selection, availability explanations, accessible mobile navigation focus handling and reduced-motion support.
- Added optional `STUDIO_MODEL_CACHE_DIR` so isolated data folders can reuse already-downloaded local models.

The broad historical backlog is still explicit: unavailable cloud databases/parsers and Bedrock need adapters and service access; the visual pipeline builder, global GraphRAG, streaming/persisted answers, reference-answer correctness, OpenTelemetry export and 10,000-document load testing are not completed by this audit. No final publication or Git push was requested.

Reproduce the compatibility matrix (from backend):

```powershell
$env:STUDIO_MATRIX_REPORT = 'D:\RAG-I\output\compatibility-sqlite.json'
..\.venv\Scripts\python.exe -m pytest tests/test_end_to_end_matrix.py -q
```

For PostgreSQL application-database checks, set `STUDIO_TEST_DATABASE_URL` to a dedicated disposable test database only. The fixture resets that database's public schema per test. `TEST_PGVECTOR_URL` separately selects the vector-store test database (default localhost:5434/vec). The audit uses dedicated application database `rag_matrix_20261105`, not the app's existing database.

The README previously pointed PostgreSQL tests at the application database. Corrected it to `studio_test` and added a fixture guard that rejects application database names before opening/resetting them. Dedicated test-target validation has eight focused regression cases; these and the two Settings tests passed (10 total). A PostgreSQL chunker/store task-matrix case also passed after adding the guard.
