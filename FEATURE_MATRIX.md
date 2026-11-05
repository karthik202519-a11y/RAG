# Feature matrix

What the RAG Learning Studio can **actually do today**, checked against running code. This file is the
source of truth: the Method Catalog in the UI is generated from the same registry
(`backend/studio/registry/`), and a test fails if the registry advertises a chunker that does not exist.

Legend: **Implemented + tested** = runs and has automated tests. **Verified live** = exercised against a real
service (Azure OpenAI, Postgres, Qdrant server, MinIO, Redis) or in the browser, without an automated test.
**Not built** = described, cannot be run.

Counts from the registry: **72+ methods registered; about 26 run, the rest are labelled "not built".**
Automated tests: **167 backend tests**, all passing on SQLite and on PostgreSQL 16. Tests never call Azure.

## Platform

| Capability | Status |
|---|---|
| Registration, login, JWT sessions, scrypt password hashing, login throttling | Implemented + tested |
| Workspaces, projects, collections; roles (owner / editor / viewer) enforced in queries | Implemented + tested (breaking the role check fails a test) |
| Cross-user isolation: unknown and forbidden both return 404 | Implemented + tested for every resource family |
| Audit events for register, project, document, index and dataset actions | Written; no viewer UI |
| Alembic migrations (0001-0003) | Implemented + tested on SQLite and PostgreSQL 16 |
| In-process job queue (thread pool): retry with backoff, dead state, cancel, restart recovery, idempotency | Implemented + tested |
| Celery + Redis job queue | **Verified live**: API and worker as separate processes, and inside Docker Compose |
| Local artifact store | Implemented + tested |
| S3-compatible artifact store (MinIO) | **Verified live**: objects confirmed in the bucket after a containerized run. No automated test |
| Server-Sent Events for live job status | Implemented; verified in the browser |
| **Docker Compose: Postgres, Redis, Qdrant, MinIO** | **Verified live**: all healthy |
| **Docker Compose `--profile app` (API, Celery worker, web UI)** | **Verified live**: images build; a full run (upload, parse on the worker, index into the Qdrant server, search, answer, evaluate) succeeded and the data was confirmed in Postgres, MinIO and Qdrant |
| Dependency completeness: every third-party import is declared in `requirements.txt` | Implemented + tested (this was a real bug the container run exposed) |
| OpenTelemetry tracing, Langfuse, secret vault, rate limiting beyond login, file scanning | Not built |

## Ingestion

| Capability | Status |
|---|---|
| Upload PDF, DOCX, **PPTX, XLSX, XML**, HTML, Markdown, TXT, CSV, JSON (multi-file, drag and drop) | Implemented + tested |
| Content-hash duplicate detection; new content under the same name becomes a new version | Implemented + tested |
| Type, size and magic-byte validation (including Office files); path stripping; XML external entities not resolved | Implemented + tested |
| Background parse job with progress, retry, cancel, manual retry; honest document state after cancel or failure | Implemented + tested |
| Tags, custom metadata, collections, filters, delete; ingestion dashboard | Implemented + tested |
| 40 documents in one request | Tested |
| **10,000-document load test** | **Not run.** No claim of large-scale readiness is made. |
| Resumable / folder upload, ZIP, images, source code, cloud connectors | Not built |

## Parsers (registry category `parser`)

| Parser | Status |
|---|---|
| PyMuPDF (text, tables, printed pages, sections, header/footer removal, diagram-page flag + PNG) | Implemented; exercised on a 186-page and a 297-page PDF |
| Plain text, Markdown, DOCX, PPTX (one page per slide), XLSX (one page per sheet), HTML, CSV, JSON, XML | Implemented + tested |
| pdfplumber, Docling, Unstructured, Tika, Tesseract, PaddleOCR, Azure Document Intelligence, AWS Textract, Google Document AI, LlamaParse | Not built |
| Side-by-side parser comparison UI | Not built |

## Chunking (registry category `chunking`): 10 run, 14 do not

| Method | Status |
|---|---|
| Fixed characters, fixed tokens, sliding window, recursive, sentence, paragraph, Markdown header, page-aware, structure-aware | Implemented + tested; all run on the 297-page NASA handbook |
| **Table-aware (row-level)**: one chunk per row (or group of rows) with caption and headers repeated; Markdown or key-value format | Implemented + tested; 492 row chunks on the handbook, none unlocated |
| Overlapping / non-overlapping | Not separate methods: the `overlap` parameter |
| HTML structure, document section, semantic, topic, parent-child, small-to-big, layout-aware, code-aware, JSON/XML structure, contextual enrichment, LLM proposition, adaptive | Not built |

Every chunk has exact character offsets, pages, printed pages and section. Measured per run: chunk count, token
distribution, duplication, coverage, mid-sentence endings, embedding tokens, optional cost, timing.

## Embeddings, indexes and retrieval

| Capability | Status |
|---|---|
| Azure OpenAI embeddings (`text-embedding-3-small`, 1,536 dims) | **Verified live**; not in automated tests |
| Offline hash embedder | Implemented + tested; **not semantic**, labelled as a demo |
| Embedding cache by (provider, model, text) | Implemented + tested; verified live (a 485-chunk rebuild billed 6k tokens, 468 cache hits) |
| Immutable index versions with a unique collection each; query/index dimension guard | Implemented + tested |
| Qdrant: embedded mode | Implemented + tested |
| Qdrant: **server mode** | **Verified live** inside Docker (collection with the right size and distance) |
| Dense, BM25 (with matched terms), hybrid RRF; filters, threshold, top-k | Implemented + tested; verified live on real data |
| **Query transforms**: rewrite, multi-query, HyDE, decomposition (each shown as a stage; HyDE never used for keyword matching) | Implemented + tested with a fake model; verified live with `gpt-4o-mini` |
| **MMR** diversification | Implemented + tested; verified live |
| **LLM reranking** with before/after rank, score, and original scores kept; failure keeps order and warns | Implemented + tested; verified live |
| Source viewer with highlighted chunk; page image for diagram pages | Implemented; verified in the browser |
| **Embedding Explorer**: 2D PCA of stored vectors, cosine/dot/Euclidean comparison, similarity matrix | Implemented + tested; verified in the browser on the real 1,536-d index |
| Grounded answers: numbered passages in a token budget, exact-prompt inspector, citation check, refusal, score gate | Implemented + tested with a fake model; verified live with `gpt-4o-mini` |
| Citation check proves the `[n]` exists, **not** that the passage supports the claim | Stated in the UI |
| **Vision descriptions of diagram pages** (`gpt-4o`), checkpointed per page, idempotent, searchable | Implemented + tested with a fake model; verified live on 4 real pages; turned a refused diagram question into a correct answer |
| Other vector stores, embedding providers, LLM providers; parent-document and small-to-big retrieval; graph retrieval | Not built |
| Streaming answers, persisted answers | Not built |

## Evaluation Center

| Capability | Status |
|---|---|
| Datasets and labelled questions (evidence = document + pages, multi-hop supported, unanswerable supported) | Implemented + tested |
| Import: studio JSON, and the prototype's two question files (questions for documents not in the project are skipped and reported) | Implemented + tested; verified live (34 NASA questions) |
| Train / validation / test splits; warning when the test split is reused | Implemented + tested |
| Hit rate, recall, precision, MRR, nDCG, MAP, keyword coverage; "not available" instead of zero when evidence is missing | Implemented + tested against hand-computed values and a 300-case randomized bounds test |
| Bootstrap 95% confidence intervals (only with 10+ questions) | Implemented + tested |
| Paired run comparison with significance, and the list of questions that improved or regressed | Implemented + tested; verified live |
| Latency p50/p95, LLM tokens, per-type breakdown, unanswerable score-gate sweep | Implemented + tested |
| Answer-level deterministic checks (correct refuse/answer, citations resolve, keyword present) | Implemented + tested |
| CSV and JSON export | Implemented + tested |
| Faithfulness, answer relevance, answer correctness (need an LLM judge or human labels) | **Not built**; the UI lists them as not available |
| LLM-generated candidate questions with human approval; human annotation UI | Not built (the `approved` flag exists) |
| Ragas / DeepEval integration | Not built |

## Not started

Parser comparison lab, GraphRAG, pipeline builder, experiment studio (beyond run comparison), traces and monitoring,
provider and connector settings UI, recommendation engine, context compression, multimodal retrieval beyond
vision descriptions, semantic / parent-child / contextual chunking, other vector databases, the 10,000-document load test.
