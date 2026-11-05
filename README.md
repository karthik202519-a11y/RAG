# RAG

A laboratory for building, comparing and understanding retrieval-augmented generation (RAG) on your own
documents. Every control is backed by code that actually runs; methods that are described but not built are
labelled **Not built yet** and cannot be run.

> **Status:** The implemented pipeline supports ingestion, parser comparison, all 23 chunking methods,
> five vector stores, dense/BM25/TF-IDF/hybrid/graph retrieval, query transforms, reranking, cited answers,
> local GraphRAG, visual tools and evaluation. Hosted adapters need their own credentials or servers.
> Pipeline Builder and Experiment Studio now save and execute versioned recipes with inspectable artifacts.
> Other explicitly unavailable catalog adapters remain documented with their prerequisites.
> See [FEATURE_MATRIX.md](FEATURE_MATRIX.md) and [BUILD_PROGRESS.md](BUILD_PROGRESS.md) for verification boundaries.

## Explore the guided workbench

Open [RAG](http://localhost:3000). Select **Guided RAG example (synthetic)** to inspect the supplied small learning example.

- **Pipeline Builder** (`/pipelines`): configure, save and execute immutable recipes.
- **Experiment Studio** (`/experiments`): compare recipes on the same frozen questions.
- **Storage Explorer** (`/storage`): inspect table purposes, schema, scoped example rows and relationships; compare real native records in all five vector stores and follow a vector back to its source chunk.
- **Learning Guide** (`/learn`): explanations for every stage, with advanced controls alongside guided defaults.

The examples demonstrate behavior; four synthetic questions do not establish retrieval quality.

## Automatic generation provider selection

Set `STUDIO_CHAT_MODEL=auto` in your local `.env`. RAG selects the first configured generation provider in this order: Google AI Studio (`GEMINI_API_KEY`), NVIDIA (`NVIDIA_API_KEY`), OpenAI (`OPENAI_API_KEY`), complete Azure OpenAI settings, then other configured providers. Selection checks configuration presence; invalid keys or exhausted quotas fail explicitly rather than silently switching services.

`STUDIO_EMBEDDING_PROVIDER=fastembed` keeps embeddings local and independent of the generating LLM. Existing providers and saved indexes remain supported. Pin a provider with `STUDIO_CHAT_MODEL=gemini:<model>`, `nvidia:<model>`, `openai:<model>`, or an Azure deployment name. An empty value preserves the previous Azure-first behavior. Restart the API after editing `.env`.

NVIDIA uses `https://integrate.api.nvidia.com/v1/chat/completions`. Google uses its `generateContent` API; the default Gemma model is configurable through `GEMINI_CHAT_MODEL`. Hosted free access is subject to account/model quotas; API-key presence does not establish free billing. Keys remain server-side in the ignored `.env` file.

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
docker compose up -d        # Postgres, pgvector, Redis, Qdrant, MinIO
```

Point the backend at them with environment variables (see `.env.example`): `DATABASE_URL=postgresql+psycopg://studio:studio_dev@127.0.0.1:5433/studio`,
`REDIS_URL=redis://127.0.0.1:6379/0` (switches jobs to Celery; run `celery -A studio.jobs.celery_app worker --pool=threads`),
`S3_ENDPOINT=http://127.0.0.1:9100` for MinIO. The dedicated pgvector store uses `PGVECTOR_URL=postgresql://studio:studio_dev@127.0.0.1:5435/rag_vectors`. Host ports 5433, 5435 and 9100/9101 avoid common conflicts on Windows.
`docker compose --profile app up -d --build` also builds and runs the API, a Celery worker and the web UI (verified end to end).

## Tests

PostgreSQL fixtures reset their target database schema. Create a dedicated test database first (`docker exec rag-studio-postgres-1 createdb -U studio studio_test`); use that database only, never the app database. The fixture enforces a database name containing `test` or `matrix`. Set `STUDIO_MODEL_CACHE_DIR` to an existing model cache to reuse local downloads.

```powershell
cd D:\RAG-I\backend
$env:STUDIO_MODEL_CACHE_DIR = 'D:\RAG-I\data\studio\models'
..\.venv\Scripts\python.exe -m pytest -q

# PostgreSQL compatibility matrix, after creating studio_test as described above:
$env:STUDIO_TEST_DATABASE_URL = 'postgresql+psycopg://studio:studio_dev@127.0.0.1:5433/studio_test'
..\.venv\Scripts\python.exe -m pytest tests/test_end_to_end_matrix.py -q
Remove-Item Env:STUDIO_TEST_DATABASE_URL

cd ..\frontend
npm run lint
npm run build
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
