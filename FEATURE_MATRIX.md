# Feature matrix

What the RAG Learning Studio can **actually do today**, checked against running code. This file is the
source of truth: the Method Catalog in the UI is generated from the same registry
(`backend/studio/registry/`), and a test fails if the registry advertises a chunker that does not exist.

Legend: **Implemented + tested** = runs and has automated tests. **Verified manually** = exercised in the
browser or against a live service, no automated test. **Not built** = described, cannot be run.

## Platform

| Capability | Status | Evidence |
|---|---|---|
| Registration, login, JWT sessions, scrypt password hashing | Implemented + tested | `tests/test_auth_projects.py` |
| Login throttling (5 failures / minute) | Implemented + tested | same |
| Workspaces, projects, collections | Implemented + tested | same |
| Role checks (owner / editor / viewer) enforced in queries | Implemented + tested | viewer test; breaking the check fails it |
| Cross-user isolation (unknown and forbidden both return 404) | Implemented + tested | `test_projects_are_isolated_between_users`, `test_documents_of_another_user_are_invisible` |
| Audit events for register, project and document actions | Implemented (written), no viewer UI | `models.AuditEvent` |
| Alembic migrations | Implemented + tested | run at start of every test; SQLite and PostgreSQL 16 |
| SQLite (default) and PostgreSQL | Implemented + tested | full suite passes on both |
| Local and S3-compatible artifact store | Local: implemented + tested. S3/MinIO: **code written, not yet exercised** | `artifacts.py` |
| In-process job queue (thread pool) | Implemented + tested | retries, backoff, cancel, restart recovery |
| Celery + Redis job queue | **Verified manually** (API and worker as separate processes on Postgres + Redis) | no automated test |
| Docker Compose infrastructure (Postgres, Redis, Qdrant, MinIO) | **Verified**: all four containers healthy | `docker-compose.yml` |
| Docker Compose application services (backend, worker, frontend) | **Not verified**: `--profile app` has no Dockerfiles yet | |
| Server-Sent Events for live job status | Implemented, verified in the browser | |
| OpenTelemetry, Langfuse, role UI, secret vault | Not built | |

## Ingestion

| Capability | Status |
|---|---|
| Upload PDF, DOCX, HTML, Markdown, TXT, CSV, JSON (multi-file, drag and drop) | Implemented + tested |
| Content-hash duplicate detection per project | Implemented + tested |
| New content under the same name becomes a new version | Implemented + tested |
| Type, size and magic-byte validation; path stripping | Implemented + tested |
| Background parse job with progress, retry/backoff, cancel, manual retry, dead state | Implemented + tested |
| Cancelled or failed parse leaves the document in an honest state | Implemented + tested |
| Jobs survive a restart (re-queued) | Implemented + tested |
| Tags, custom metadata, collections, filters, delete | Implemented + tested |
| Ingestion dashboard (documents, pages, tokens, storage, throughput, success rate, queue depth) | Implemented + tested |
| 40 documents in one request | Tested |
| **10,000-document load test** | **Not run.** No claim of large-scale readiness is made. |
| Resumable / multipart upload, folder upload | Not built |
| PPTX, XLSX, XML, images, source code, ZIP | Not built |
| Cloud connectors (Blob, S3, GCS, Drive, SharePoint, URL, Git, Confluence) | Not built |
| Pause / resume jobs | Not built (cancel and retry exist) |

## Parsers (registry category `parser`)

| Parser | Status |
|---|---|
| PyMuPDF (text blocks, tables to Markdown, printed page numbers, section titles, header/footer removal, diagram-page flag + PNG) | Implemented, exercised on two real 186- and 297-page PDFs |
| Plain text, Markdown, DOCX (python-docx), HTML (BeautifulSoup), CSV, JSON | Implemented + tested |
| pdfplumber, Docling, Unstructured, Tika, Tesseract, PaddleOCR, Azure Document Intelligence, AWS Textract, Google Document AI, LlamaParse, PPTX, XLSX | Not built (listed as unavailable with a reason) |
| Side-by-side parser comparison UI | Not built |

## Chunking (registry category `chunking`): 9 run, 15 do not

| Method | Status |
|---|---|
| Fixed characters, fixed tokens, sliding window, recursive, sentence, paragraph, Markdown header, page-aware, structure-aware | **Implemented + tested**; run on the 297-page NASA handbook |
| Overlapping / non-overlapping | Not separate methods: the `overlap` parameter (0 = none) |
| HTML structure, document section, semantic, topic, parent-child, small-to-big, layout-aware, table-aware, code-aware (tree-sitter), JSON/XML structure, contextual enrichment, LLM proposition, adaptive | Not built |

Parameter validation (ranges, overlap < size, stride <= size, unknown parameters) is tested. Measured per
run: chunk count, token distribution, duplication, coverage, mid-sentence endings, embedding tokens,
optional cost (needs a price you enter), timing.

## Embeddings, indexes and retrieval (slice 2)

| Capability | Status |
|---|---|
| Azure OpenAI embeddings (`text-embedding-3-small`, 1,536 dims) | **Verified live** on real documents: 481 chunks, 184k tokens billed. Not covered by automated tests (tests never call Azure). |
| Offline hash embedder (`local_hash`) | Implemented + tested. **Not semantic**; labelled as a demo in the UI and registry. |
| Embedding cache keyed by (provider, model, text), shared across indexes | Implemented + tested (rebuild billed 0 tokens, 481 cache hits, verified live) |
| Immutable index versions: frozen chunker, documents, model; unique collection per index | Implemented + tested |
| Qdrant vector store: embedded (on-disk) mode | Implemented + tested |
| Qdrant server mode (`QDRANT_URL`) | Code written, container healthy, **never exercised by the app** |
| Query/index dimension guard (409 instead of searching mismatched vectors) | Implemented + tested |
| Dense (vector) retrieval | Implemented + tested; live: 17/23 rank-1, 20/23 top-5 on the 23 labelled NASA questions |
| BM25 keyword retrieval with matched terms | Implemented + tested; live: 8/23 rank-1, 17/23 top-5 |
| Hybrid retrieval with reciprocal rank fusion and per-retriever contributions | Implemented + tested; live: 13/23 rank-1, 20/23 top-5 (**did not beat dense on these questions**) |
| Filters (document, chunk type), score threshold, top-k, candidates, RRF k | Implemented + tested |
| Source viewer: highlight the chunk in the original text; page image for diagram pages | Implemented, verified in the browser |
| Retrieval runs recorded; stored-record browser; stored-vector inspector | Implemented + tested |
| Compare-modes view | Implemented, verified in the browser |
| Other vector stores (pgvector, Chroma, FAISS, Azure AI Search, Pinecone, Milvus, ...) | Not built |
| Other embedding providers (OpenAI, Sentence Transformers, Ollama, Cohere, Voyage, CLIP) | Not built |
| MMR, multi-query, HyDE, query rewriting, parent-document, reranking, graph retrieval | Not built |
| Embedding Explorer (similarity matrix, 2D projection, model comparison) | Not built (the building blocks, stored vectors and an inspector, exist) |
| Answer generation (Azure OpenAI chat): grounded prompt, token-budgeted context, numbered citations, exact-prompt inspector | Implemented + tested with a fake model; **verified live** with `gpt-4o-mini` on the real index (answer, citation check, refusal) |
| Citation check: every `[n]` must point at a passage that was sent | Implemented + tested. Proves the number exists, **not** that the passage supports the claim |
| Abstention: grounded-mode refusal, and a minimum-score gate that skips the model call (0 tokens) | Implemented + tested; verified live |
| Chat deployments `gpt-4o-mini`, `gpt-4o`, `gpt-5-mini`, `gpt-5` | Probed and exist on the configured resource; only `gpt-4o-mini` exercised in an answer |
| Streaming answers, persisted answers, faithfulness scoring, other LLM providers | Not built |
| Recall/precision/nDCG, faithfulness, experiment studio | Not built; hit@k/MRR was computed by a script, not by the studio |

## Not started

Table and multimodal labs, GraphRAG, context compression, generation and prompt lab,
pipeline builder, experiment studio, evaluation center, traces, recommendation engine, provider and
connector settings. The earlier prototype under `app/` and `scripts/` holds the 48-question evaluation set, which the studio
does not use yet.
