# Feature matrix

What RAG can **actually do today**, checked against running code. This file is the
source of truth: the Method Catalog in the UI is generated from the same registry
(`backend/studio/registry/`), and a test fails if the registry advertises a chunker that does not exist.

Legend: **Implemented + tested** = runs and has automated tests. **Verified live** = exercised against a real
service (Azure OpenAI, Postgres, Qdrant server, MinIO, Redis) or in the browser, without an automated test.
**Not built** = described, cannot be run.

Counts from the registry (93 methods): **chunking 23 of 23 run; parsers 15 of 23; embedding 8 of 9; vector stores 5 of 11; retrieval 14 of 14; generation 5 of 6; multimodal 6 of 7.** Anything that cannot run says exactly why (a missing server, key or heavy dependency), and hosted-service adapters are marked as tested against simulated responses only.
Automated tests are recorded by the current audit in `BUILD_PROGRESS.md` and the generated JUnit / compatibility reports under `output/`. Historical test counts are not current-suite evidence. Hosted API transports are simulated in automated tests.

## Storage Explorer

Read-only `/storage` shows all 23 application tables with schemas, purpose, benefits, project-scoped examples and relationships. Foreign keys are distinguished from logical IDs/hashes/JSON links. Native storage reads actual Qdrant points, FAISS rows and companion metadata, Chroma records, Lance tables and pgvector SQL rows with actual primary-key/HNSW definitions. Vector IDs link back to source chunks; passwords are redacted and foreign project samples are excluded. Seven tests passed on each application database, covering every real vector store. All five engines and record navigation were also checked in the browser.

## Platform

| Capability | Status |
|---|---|
| Registration, login, JWT sessions, scrypt password hashing, login throttling | Implemented + tested |
| Workspaces, projects, collections; roles (owner / editor / viewer) enforced in queries | Implemented + tested (breaking the role check fails a test) |
| Cross-user isolation: unknown and forbidden both return 404 | Implemented + tested for every resource family |
| Audit events for register, project, document, index and dataset actions | Written; scoped samples in Storage Explorer |
| Alembic migrations (0001-0007) | Implemented + tested on SQLite and PostgreSQL 16 |
| In-process job queue (thread pool): retry with backoff, dead state, cancel, restart recovery, idempotency | Implemented + tested |
| Celery + Redis job queue | **Verified live**: API and worker as separate processes, and inside Docker Compose |
| Local artifact store | Implemented + tested |
| S3-compatible artifact store (MinIO) | **Verified live**: objects confirmed in the bucket after a containerized run. No automated test |
| Server-Sent Events for live job status | Implemented; verified in the browser |
| **Docker Compose: Postgres, Redis, Qdrant, MinIO** | **Verified live**: all healthy |
| **Docker Compose `--profile app` (API, Celery worker, web UI)** | **Verified live**: images build; a full run (upload, parse on the worker, index into the Qdrant server, search, answer, evaluate) succeeded and the data was confirmed in Postgres, MinIO and Qdrant |
| Dependency completeness: every third-party import is declared in `requirements.txt` | Implemented + tested (this was a real bug the container run exposed) |
| **Traces page**: every search, answer and job with its measured steps as a waterfall, p50/p95, token totals | Implemented + tested; verified in the browser. Spans are the stage timings the code recorded, not OpenTelemetry |
| **Settings page**: runtime configuration, which secrets are set (never their values), live connection tests (database, storage, Qdrant, FAISS, embedding, chat, vision) | Implemented + tested; embedding and chat tests verified live against Azure |
| OpenTelemetry export, Langfuse, secret vault, rate limiting beyond login, file scanning | Not built |

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
| Images and source-code upload | Implemented + tested (see parser and visual content below) |
| Resumable / folder upload, ZIP, cloud connectors | Not built |

## Parsers (registry category `parser`)

| Parser | Status |
|---|---|
| PyMuPDF (text, tables, printed pages, sections, header/footer removal, diagram-page flag + PNG) | Implemented; exercised on a 186-page and a 297-page PDF |
| Plain text, Markdown, DOCX, PPTX (one page per slide), XLSX (one page per sheet), HTML, CSV, JSON, XML | Implemented + tested |
| **Parser Lab**: PyMuPDF, pdfplumber and pypdf on the same pages side by side, with text, table counts, timings and word-level agreement | Implemented + tested; verified on the handbook (pages 120 to 122: pdfplumber found 6 tables, PyMuPDF 0). Ingestion also supports selecting pdfplumber, RapidOCR or Azure Document Intelligence |
| Azure Document Intelligence | Adapter implemented; simulated-response tests only |
| Docling, Unstructured, Tika, Tesseract, PaddleOCR, AWS Textract, Google Document AI, LlamaParse | Not built |

## Chunking (registry category `chunking`): all 23 run (2 need the chat model)

| Method | Status |
|---|---|
| Fixed characters, fixed tokens, sliding window, recursive, sentence, paragraph, Markdown header, page-aware, structure-aware | Implemented + tested; run on the 297-page handbook and on 8.4 million characters of real files |
| Table-aware (row-level), semantic (embedding-based), parent-child (small-to-big) | Implemented + tested |
| **Overlapping** (percentage preset) and **non-overlapping** (recursive or fixed, overlap forced to 0) | Implemented + tested |
| **HTML structure** (headings, lists, tables), **section-based**, **topic-based** (TextTiling, no model), **layout-aware** (font size, columns, gaps from PDF layout data), **code-aware** (Python syntax tree; declaration patterns for other languages), **JSON/XML structure** (exact record slices) | Implemented + tested on matching inputs; each falls back to recursive splitting, and says so, on documents it is not designed for |
| **Contextual** (a chat model writes one stored sentence per chunk) and **proposition** (a chat model rewrites passages into atomic facts, each kept only if its words occur in the source) | Implemented + tested with a fake model; verified live on `gpt-4o-mini` (40 chunks enriched in 32 s; 12 passages into 160 propositions in 19 s) |
| **Adaptive**: measures the document and picks the method, printing the measurements and the reason | Implemented + tested; the same rules power the Suggestions page |

Layout-aware needs position and font data, which is recorded for PDFs uploaded after this feature; older uploads fall back and are told to re-upload.
Every chunk has exact character offsets, pages, printed pages and section (except proposition chunks, which are the model's wording and have no highlight). Measured per run: chunk count, token distribution, duplication, coverage, mid-sentence endings, embedding tokens, optional cost, timing.

## Embeddings, indexes and retrieval

| Capability | Status |
|---|---|
| Azure OpenAI embeddings (`text-embedding-3-small`, 1,536 dims) | **Verified live**; not in automated tests |
| Offline hash embedder | Implemented + tested; **not semantic**, labelled as a demo |
| Embedding cache by (provider, model, text), with separate query/document roles for Cohere and Voyage | Implemented + tested; verified live (a 485-chunk rebuild billed 6k tokens, 468 cache hits) |
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
| **FAISS vector store** (exact, flat; filters run inside the search), chosen per index; same top results as Qdrant in tests | Implemented + tested; connection test verified in the browser |
| **GraphRAG slice**: a model extracts entities and relations per chunk; anything not found in the chunk text is rejected; communities by modularity; local graph search compared with dense search; interactive graph | Implemented + tested with a fake model; **verified live** on 60 handbook chunks (534 entities, 139 relations, 445 model proposals rejected as not in the text, 332 entities with no relations). Not evaluated for answer quality; no global (community-summary) search |
| **Vector stores**: Qdrant, FAISS, **Chroma**, **LanceDB**, **pgvector** behind one interface, with one contract test run against all five | Implemented + tested; pgvector tested live in the `pgvector/pgvector:pg16` container |
| Milvus, Weaviate, Elasticsearch/OpenSearch, Azure AI Search, Pinecone, MongoDB Atlas | Not built: each needs a server or cloud account (reason shown in the catalog) |
| **Embeddings**: Azure OpenAI, **FastEmbed local semantic models** (BGE default), OpenAI, Ollama, Cohere, Voyage | FastEmbed tested on the real model; the others tested with simulated responses only (no keys or servers here) |
| **Cross-encoder reranking** (local, free) beside LLM reranking | Implemented + tested on the real model |
| **Chat providers**: Azure OpenAI plus `openai:`, `anthropic:`, `gemini:`, `ollama:` prefixed models | Adapters tested with simulated responses only; AWS Bedrock not built |
| Streaming answers, persisted answers | Not built |

## Suggestions (mandatory stage advice)

| Item | Status |
|---|---|
| **Suggestions page**: a pick for parsing, chunking, embedding, vector store, retrieval, reranking, generation and evaluation, each with reasons, evidence and alternatives, labelled **Measured** (from your evaluation runs, only when the gap is outside the confidence interval on 10+ questions), **Rule** (from measurements of your documents) or **Starting point** | Implemented + tested |
| Per-document suggestion chip in the Chunking Playground ("Suggested: Code-aware, because the file is source code") with a Use button | Implemented + tested |
| Next experiments to run (missing baselines, FAISS exact check, reranker on/off, contextual on a subset) | Implemented + tested |

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
| **LLM judge**: splits an answer into claims and checks each against the passages; a claim counts as supported only if the quote the judge gives really occurs in the passage. Per-answer button in the Retrieval Lab and an optional evaluation-run setting | Implemented + tested with a fake model; verified live (faithful answer 1.0; answer with a wrong and an invented claim 0.33). A model's opinion, not ground truth |
| **Experiment grid**: run every combination of indexes, modes, k and rerankers on the same questions (up to 24 runs) | Implemented + tested; compare results with the existing paired comparison |
| Answer correctness (needs reference answers) | **Not built**; the UI lists it as not available |
| LLM-generated candidate questions with human approval; human annotation UI | Not built (the `approved` flag exists) |
| Ragas / DeepEval integration | Not built |

## Performance (measured on the dev machine, Azure `text-embedding-3-small`)

| Item | Status |
|---|---|
| Embedding batches run in parallel with a content-hash cache | Implemented + tested; about 17 to 46 texts/s on the same corpus |
| Evaluation and vision-description jobs run questions/pages with a worker pool | Implemented + tested |
| GZip responses, cached method catalog, route prefetch on link hover, lazy-loaded charts | Implemented; first-load JS about 200 KB gzip; pages show their heading in 170 to 350 ms locally |

## Parsers and visual content (added)

| Item | Status |
|---|---|
| **Parser choice on upload**: PyMuPDF (default), **pdfplumber** (ruled tables), **RapidOCR** (local OCR for scanned PDFs and PNG/JPEG), **Azure Document Intelligence** | pdfplumber and OCR tested on real files; Azure tested with a recorded response only |
| **Source code upload** (Python, JS/TS, Java, Go, Rust, C/C++, C#, Ruby, PHP, shell, SQL, Kotlin, Swift) | Implemented + tested |
| **Visual Lab**: figure crops (pictures and drawing clusters), chart-to-table with the vision model (flagged as model-read), **CLIP page search** (local ONNX, find pages by appearance) | Implemented + tested; CLIP tested on the real model |
| Docling, Unstructured, Tika, Tesseract, PaddleOCR, AWS Textract, Google Document AI, LlamaParse | Not built; reasons in the catalog (heavy dependencies, Java, missing binaries, cloud accounts) |

## Scale test (see docs/SCALE_TEST.md)

| Item | Status |
|---|---|
| Every model-free chunking method on corpora of 0.06M, 0.86M, 3.3M and 8.4M characters of real files | **Measured**: each method finishes the 8.4M corpus in 2 to 24 seconds at under 165 MB |
| End-to-end through the API on a 10,000-character corpus (upload, parse, Azure embeddings, needle questions) | **Measured**: 5 files, 0 failures, 129 chunks, 26 questions, hit 0.96 |
| End-to-end at 1M, 3M and 10M characters | **Not run**: stopped at your request; the scripts exist (`scripts/scale_bench/`) |
| Vector-store comparison, semantic and LLM chunker timings | **Measured** on the 10,000-character corpus's vectors and a 300k-character sample |

## Not started

OpenTelemetry export, context compression, global GraphRAG search with
community summaries, the vector stores and parsers listed as not built above, the 10,000-document load test.


## Guided workbench, pipeline recipes and controlled experiments (2026-11-05)

| Capability | Current implementation |
|---|---|
| Pipeline Builder | Selectable typed stage canvas, four starting templates, all chunkers/providers/stores, document/index reuse, immutable saved versions, validation, background execution, real stage timings and artifacts, cancellation and JSON export |
| Experiment Studio | Two to eight saved versions, frozen approved-question snapshot, isolated index per case, quality/latency/tokens, paired differences, question drill-down, saved executions and JSON report export |
| Search methods | Six base retrievers: dense, BM25, TF-IDF, hybrid, local graph, graph + hybrid. Fifteen search recipes expose transforms, MMR, parent/neighbor expansion and both rerankers in one selector |
| Explanations | Guided explanations default open; method-specific algorithm, advantages, tradeoffs and prerequisites; a complete Learning Guide route |
| Graph construction | Actual extraction model and counters, retained entity/relation examples with source chunk text; grounding limitations stated explicitly |
| Embedding construction | Actual embedding input, stored vector preview, model/store/dimension/cache/tokens, and a PCA construction explanation |
| Advanced answers | Generation and evaluation use the exact transformed/reranked/expanded retrieval output rather than silently repeating base search |

The canvas supports the ordered typed RAG stages; arbitrary user-defined graph nodes or custom executable code are not exposed. Graph extraction in pipeline/experiment runs samples at most 50 chunks, with model calls disclosed in the UI. The seeded example is explicitly synthetic and too small for significance claims.
