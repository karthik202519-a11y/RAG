# RAG-I

A step-by-step, evaluated RAG **retrieval** pipeline for long PDFs that contain text, tables,
and figures. Upload a PDF, chunk it, embed the chunks, and search with cited results
(document, PDF page, printed page). Everything is an HTTP API.

> Status: retrieval works end to end and is measured; ranking, charts and a vector database
> are still to come. See [docs/STATUS.md](docs/STATUS.md) for what is done and what is pending.

```
PDF --upload--> extract --> chunk --> embed --> search
        POST /documents   /chunk   /embed   POST /search
```

## Quick start

```bash
git clone https://github.com/karthik202519-a11y/RAG-I.git
cd RAG-I
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then fill in your Azure OpenAI details
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive API.

No Azure account yet? Set `RAG_EMBEDDING_BACKEND=fake` in `.env`. It uses an offline hashing
embedder: good enough to try the API, not for real retrieval quality.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents` | Upload a PDF (multipart `file`). Parses it unless `?extract=false`. Same bytes give the same `doc_id`. |
| `POST` | `/documents/{id}/extract` | (Re)parse the PDF into pages of text blocks and tables |
| `GET` | `/documents`, `/documents/{id}` | List documents / one document with its status |
| `DELETE` | `/documents/{id}` | Delete a document and everything derived from it |
| `GET` | `/documents/{id}/pages/{n}` | The extracted content of page *n* (what the chunker sees) |
| `GET` | `/documents/{id}/pages/{n}/image` | PNG of a diagram or chart page |
| `POST` | `/documents/{id}/chunk` | Chunk at `{"size": 400}` tokens (optional `overlap`) |
| `GET` | `/documents/{id}/chunks` | Browse chunks (`size`, `offset`, `limit`, `type`) |
| `POST` | `/documents/{id}/embed` | Embed all chunks of a size |
| `POST` | `/documents/{id}/process` | Extract (if needed), chunk and embed in one call |
| `POST` | `/search` | Top-k chunks for a query, with scores and citations |
| `GET` | `/health` | Status and active embedding backend |

### Example

```bash
# 1. upload and parse
curl -F "file=@handbook.pdf;type=application/pdf" http://127.0.0.1:8000/documents
#    -> {"doc_id": "8eeb4887a4dc", "pages": 297, "tables": 40, ...}

# 2. chunk and embed
curl -X POST http://127.0.0.1:8000/documents/8eeb4887a4dc/process \
     -H "Content-Type: application/json" -d '{"size": 400}'

# 3. search
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" \
     -d '{"query": "What are the entrance criteria for the Preliminary Design Review?", "top_k": 3}'
```

```json
{
  "results": [
    {
      "rank": 1, "score": 0.6388, "type": "table",
      "citation": "handbook.pdf, PDF p.171 (printed 161)",
      "section": "6.0 Crosscutting Technical Management",
      "pages": [171], "text": "TABLE 6.7-1 Purpose and Results for Life-Cycle Reviews ..."
    }
  ]
}
```

Search covers all documents by default; restrict with `doc_ids`, `types` (`text`/`table`)
or `chunk_size`. Documents not yet embedded at that size are listed under `skipped`,
not silently ignored.

## How it works

- **Extraction** (`app/extraction.py`): PyMuPDF text blocks in the PDF's own reading order,
  tables rebuilt cell by cell into Markdown, repeated headers/footers removed, printed page
  numbers and section titles captured. Pages with many vector drawings are flagged
  `needs_vision` and rendered to PNG.
- **Chunking** (`app/chunking.py`): structure-aware. Prose blocks are grouped up to a token
  budget with block-level overlap. Tables stay whole with their caption and notes; long
  tables split by rows with the header repeated; tables continued on the next page are linked.
- **Embedding** (`app/embeddings.py`): Azure OpenAI `text-embedding-3-small`, with a
  content-hash cache so a text is billed once across documents and re-runs.
- **Retrieval** (`app/retrieval.py`): cosine similarity over the stored vectors.
- **Storage** (`app/store.py`): plain files under `data/documents/<doc_id>/`
  (`source.pdf`, `pages.json`, `chunks_<size>.json`, `vectors_<size>.npy`).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

Tests are offline (fake embedder) and build their own small PDFs.

## Research scripts

`scripts/` holds the numbered scripts used to study each stage and to measure it
(`07_eval_retrieval.py`, `08_eval_hard.py`, questions in `eval/`). They read a `output/`
folder produced by the earlier command-line version of this pipeline and are kept as the
record of how the choices were made. The `app/` package is the maintained code.

## Notes

- **Extraction is slow with PyMuPDF 2.0**: about 6.5 minutes for a 297-page handbook, and the
  upload request blocks until it finishes. Use `?extract=false` and call `/extract` separately
  if your client has short timeouts. A background job queue is on the list.
- **No authentication.** Run it locally or behind a gateway.
- Your PDFs and `.env` are git-ignored. Bring your own documents.

## License

MIT, see [LICENSE](LICENSE).
