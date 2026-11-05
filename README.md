# RAG Learning Studio

A laboratory for building, comparing and understanding retrieval-augmented generation (RAG) on your own
documents. Every control is backed by code that actually runs; methods that are described but not built are
labelled **Not built yet** and cannot be run.

> **Status: early.** Ingestion, the chunking lab, embeddings, Qdrant indexes and retrieval (dense, BM25, hybrid) work.
> Grounded answers with checked citations, query transforms, reranking, vision descriptions of diagrams, an Embedding
> Explorer and an Evaluation Center work too. The parser lab, GraphRAG, pipeline builder and several others are not built. See [FEATURE_MATRIX.md](FEATURE_MATRIX.md) for exactly
> what runs and [BUILD_PROGRESS.md](BUILD_PROGRESS.md) for what was verified and what is outstanding.

## What you can do today

- Sign up, create projects and collections.
- Upload PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, CSV, JSON and XML files (drag and drop, many at once). Duplicates are
  detected by content hash; changed content becomes a new version.
- Watch background parse jobs with live progress; cancel, retry, and see failures with their reason.
- **Chunking Playground**: cut a real document with 9 methods (fixed characters/tokens, sliding window, recursive,
  sentence, paragraph, Markdown header, page-aware, structure-aware with table captions), see exactly where each chunk
  starts and ends, mark overlap, and compare up to four configurations side by side with measured statistics.
- **Indexes & Embeddings**: build an immutable index (documents + chunker + embedding model) in Qdrant, with Azure OpenAI
  embeddings (or an offline demo embedder), an embedding cache, and a stored-record and vector inspector.
- **Retrieval Lab**: dense, BM25 and hybrid search with the contribution of each retriever on every result, query-embedding
  details, filters, source highlighting in the original document, and a "compare modes" view.
- **Answers with citations**: ask a chat model (your Azure deployments) to answer only from the retrieved passages; see the exact
  prompt, token use, and whether each `[n]` citation points at a passage that was sent. A score gate can refuse without spending tokens.
- **Evaluation Center**: label questions with where the answer lives, run them against any index or retrieval setup, and compare runs
  on the same questions with confidence intervals and a paired significance check. Metrics without evidence show "not available".
- **Embedding Explorer**: a 2D projection of your stored vectors and a side-by-side cosine / dot / Euclidean comparison.
- **Method Catalog**: 70+ methods with explanations, parameters, advantages, disadvantages and honest status.

## Quick start (no Docker needed)

Backend (Python 3.12+; the project is tested on 3.14):

```bash
cd backend
python -m venv ../.venv && source ../.venv/bin/activate     # Windows: ..\.venv\Scripts\activate
pip install -r requirements.txt
PYTHONPATH=. uvicorn studio.main:app --port 8000             # Windows PowerShell: $env:PYTHONPATH="."
```

Frontend (Node 20+):

```bash
cd frontend
npm install
npm run dev            # http://localhost:3000
```

It starts on SQLite, a local artifact folder and an in-process job pool. API docs: <http://127.0.0.1:8000/docs>.

## Scale profile (Docker)

```bash
docker compose up -d        # Postgres, Redis, Qdrant, MinIO
```

Point the backend at them with environment variables (see `.env.example`): `DATABASE_URL=postgresql+psycopg://studio:studio_dev@127.0.0.1:5433/studio`,
`REDIS_URL=redis://127.0.0.1:6379/0` (switches jobs to Celery; run `celery -A studio.jobs.celery_app worker --pool=threads`),
`S3_ENDPOINT=http://127.0.0.1:9100` for MinIO. Host ports 5433 and 9100/9101 avoid common conflicts on Windows.
`docker compose --profile app up -d --build` also builds and runs the API, a Celery worker and the web UI (verified end to end).

## Tests

```bash
cd backend && pip install -r requirements-dev.txt
PYTHONPATH=. pytest tests -q                                   # SQLite
STUDIO_TEST_DATABASE_URL=postgresql+psycopg://studio:studio_dev@127.0.0.1:5433/studio PYTHONPATH=. pytest tests -q   # Postgres
cd ../frontend && npx tsc --noEmit && npx eslint src
```

## Layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). In short: `backend/studio` (API, registry, parsing, chunking, jobs),
`frontend` (Next.js), `docker-compose.yml`, and the earlier command-line prototype in `app/` and `scripts/`
(Azure embeddings, dense retrieval, a 48-question evaluation set), which the studio will absorb in the next slice.

## Notes

- Parsing PDFs with PyMuPDF 2.0 is slow (about 1-6 s/page depending on load). Progress is reported per page.
- No authentication hardening beyond JWT + scrypt + login throttling. Run locally or behind a gateway.
- Your documents, `.env` and `data/` are git-ignored.

## License

MIT, see [LICENSE](LICENSE).
