# Status: where we are, and what is pending for retrieval

Scope of this document: **retrieval only** (find the right cited passages). Generating an
answer with an LLM comes after retrieval is good enough.

## Built and working

| Stage | What it does | Where |
|---|---|---|
| Upload | Accept a PDF, content-hash it (same file = same id) | `POST /documents` |
| Extract | Text blocks in reading order, tables to Markdown, printed page numbers, section titles, header/footer removal, ligature/control-character clean-up, diagram pages flagged and rendered to PNG | `app/extraction.py` |
| Chunk | Structure-aware, block-level overlap, tables kept whole (caption and notes attached, long tables split by rows with repeated header, continued tables linked), tiny chunks merged | `app/chunking.py` |
| Embed | Azure `text-embedding-3-small` (1536 dims), hash-keyed cache so text is billed once | `app/embeddings.py` |
| Retrieve | Cosine similarity over all documents, filters by document and chunk type, citation string with PDF and printed page | `POST /search` |
| Tests | 15 offline API tests, CI on every push | `tests/` |

## Measured quality (two PDFs: NASA handbook 297 pages, IPCC AR6 SYR 186 pages)

| Test set | Result |
|---|---|
| First round, 23 questions, 400-token chunks | answer page in top 5: 20 of 23; ranked first: 17 of 23 |
| Hard set, 22 answerable questions across both documents | answer page in top 5: 18 of 22; **ranked first: 9 of 22** |
| Off-topic question | top score about 0.08, against 0.50-0.68 for real matches, so a threshold can work |

Chunk size 400 tokens is the default. 200 and 800 were within noise of it.

## Pending for retrieval, most valuable first

| # | Item | Why | Notes |
|---|---|---|---|
| 1 | **Hybrid search** (BM25 keyword + vector) | Exact terms (acronyms, "TABLE 6.7-1", "PDR") are weak spots for embeddings | No new service needed |
| 2 | **Reranker** | Right page often retrieved but not first (9 of 22 first) | Cross-encoder, or an LLM reranker on Azure |
| 3 | **Relevance threshold** ("no answer") | Today any query returns its nearest chunks, even off-topic | Needs calibrating on unanswerable questions |
| 4 | **Charts and diagrams** | Scatter dots, heatmap cells and arrows are drawn shapes with no text, so their data is not searchable | Needs a vision-capable model to describe figure crops; flagged pages are already rendered. The flag also misses image-built figures (NASA p61) |
| 5 | **Row-level table chunks** | A 12-column grid is one vector and retrieves poorly | Row plus headers, and a table summary |
| 6 | **Multi-hop questions** | Answers split across pages or documents | Query decomposition |
| 7 | **IPCC table captions** | IPCC does not use "Table N" captions, so only 1 of 12 tables got one | Extend caption detection |
| 8 | **Mathematical formulas and other content types** | Not reviewed yet | Next in the review order |
| 9 | **Vector database** | Search is brute-force NumPy, fine for thousands of chunks | Compare pgvector, Qdrant, Chroma, Pinecone on identical chunks and questions |
| 10 | **Evaluation inside the repo** | Eval scripts still read the research `output/` folder | Turn into a command that runs against the API |
| 11 | **Background extraction** | Extraction is a blocking request: about 6.5 minutes for 297 pages | Job queue with status polling |
| 12 | **Auth and limits** | None. Run locally or behind a gateway | Before any public deployment |

## Known extraction limits

- Charts, maps and flowchart arrows: only labels survive (see item 4).
- Table detection is engine-dependent; PyMuPDF is pinned for that reason.
- Pages with a garbled figure (NASA p28) extract badly.
- Two-column reading order relies on the PDF's own content order, which was right on every page checked, but is not guaranteed for other PDFs.
