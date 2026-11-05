# Scale test: 1 to 10 million characters

Real text from this machine (Python, JavaScript and TypeScript sources, Markdown, text, JSON, CSV, HTML and the two real PDFs), in a fixed mix, taken in a fixed order so each corpus contains the smaller one. Nothing is generated. Reproduce with `scripts/scale_bench/` (see the end of this file). The numbers below are written by `make_report.py` from `docs/scale/*.json`.

## 1. Chunking: every method that needs no model, on every corpus

- **10k corpus**: 6 documents, 58,779 characters; mix prose 25%, code 22%, data 3%, web 47%, docs 3%
- **1M corpus**: 99 documents, 857,020 characters; mix prose 19%, code 47%, data 17%, web 5%, docs 12%
- **3M corpus**: 240 documents, 3,300,739 characters; mix prose 41%, code 36%, data 11%, web 3%, docs 9%
- **10M corpus**: 691 documents, 8,369,356 characters; mix prose 22%, code 47%, data 14%, web 4%, docs 12%

Seconds to chunk the whole corpus (single process, one document at a time), chunks produced, and how often the method fell back to a plain splitter because the document was not the kind it is designed for (for example HTML-structure on a Python file).

| Method | 10k s | 10k chunks | 1M s | 1M chunks | 3M s | 3M chunks | 10M s | 10M chunks | Mchar/s at 10M | Mean tokens | Coverage | Fell back (docs, 10M) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_char | 0.0 | 237 | 0.7 | 3,436 | 0.6 | 13,271 | 1.8 | 33,582 | 4.62 | 96 | 1.00 | 0 of 691 |
| fixed_token | 0.0 | 44 | 1.3 | 613 | 1.4 | 2,278 | 3.7 | 5,882 | 2.27 | 378 | 1.00 | 0 of 691 |
| sliding_window | 0.0 | 73 | 0.9 | 988 | 1.6 | 3,810 | 4.8 | 9,736 | 1.76 | 393 | 1.00 | 0 of 691 |
| recursive | 0.1 | 46 | 2.6 | 631 | 4.5 | 2,360 | 15.5 | 6,062 | 0.54 | 349 | 1.00 | 0 of 691 |
| sentence | 0.1 | 47 | 1.1 | 661 | 3.1 | 2,399 | 9.2 | 6,256 | 0.90 | 341 | 1.00 | 0 of 691 |
| paragraph | 0.1 | 49 | 0.8 | 677 | 3.3 | 2,510 | 11.2 | 6,424 | 0.74 | 338 | 1.00 | 0 of 691 |
| markdown_header | 0.1 | 58 | 1.6 | 1,132 | 4.5 | 2,711 | 14.7 | 7,368 | 0.57 | 288 | 1.00 | 457 of 691 |
| page_aware | 0.1 | 23 | 0.8 | 274 | 2.5 | 1,102 | 7.2 | 2,646 | 1.17 | 762 | 1.00 | 0 of 691 |
| structure_aware | 0.1 | 46 | 1.0 | 468 | 2.6 | 1,888 | 5.7 | 4,579 | 1.47 | 447 | 0.87 | 0 of 691 |
| table_rows | 0.0 | 58 | 0.8 | 480 | 2.1 | 2,324 | 6.0 | 5,015 | 1.39 | 412 | 0.87 | 0 of 691 |
| parent_child | 0.1 | 89 | 3.1 | 1,236 | 8.9 | 4,698 | 24.0 | 12,068 | 0.35 | 184 | 1.00 | 0 of 691 |
| overlapping | 0.0 | 46 | 0.4 | 640 | 1.6 | 2,397 | 3.7 | 6,168 | 2.27 | 380 | 1.00 | 0 of 691 |
| non_overlapping | 0.1 | 44 | 2.2 | 606 | 4.2 | 2,267 | 11.6 | 5,832 | 0.72 | 346 | 1.00 | 0 of 691 |
| html_structure | 0.1 | 53 | 1.6 | 663 | 4.0 | 2,586 | 11.7 | 7,275 | 0.71 | 277 | 1.00 | 310 of 691 |
| section_based | 0.1 | 54 | 0.9 | 731 | 3.0 | 2,861 | 7.5 | 7,195 | 1.11 | 280 | 1.00 | 0 of 691 |
| topic_based | 0.1 | 160 | 0.7 | 1,245 | 2.6 | 5,317 | 6.2 | 11,709 | 1.35 | 172 | 1.00 | 0 of 691 |
| layout_aware | 0.1 | 44 | 1.6 | 609 | 4.3 | 2,857 | 11.2 | 6,422 | 0.74 | 314 | 1.00 | 689 of 691 |
| code_aware | 0.1 | 44 | 1.5 | 650 | 4.5 | 2,391 | 13.2 | 6,234 | 0.63 | 790 | 1.00 | 225 of 691 |
| json_xml_structure | 0.1 | 44 | 1.3 | 619 | 4.1 | 2,279 | 11.6 | 5,885 | 0.72 | 342 | 1.00 | 592 of 691 |
| adaptive | 0.1 | 60 | 1.8 | 857 | 4.5 | 3,562 | 12.2 | 9,003 | 0.69 | 549 | 1.00 | 36 of 691 |

Peak memory of the benchmark process at the largest corpus: 162 MB.
Failed documents: 1M: structure_aware failed on 2 document(s) (error text was not captured in that run); 1M: table_rows failed on 2 document(s) (error text was not captured in that run). These did not recur on the larger corpora, which contain the same documents, and the cause was not identified.

## 2. Chunkers that call a model

- **semantic_fastembed_100k** (local fastembed, free): 99,586 characters, 675 sentences embedded in 127.1 s (5.3 sentences/s) into 75 chunks. Extrapolated to 10M characters: about 213 minutes.
- **semantic_fastembed_300k** (local fastembed, free): 295,042 characters, 2,122 sentences embedded in 191.7 s (11.1 sentences/s) into 201 chunks. Extrapolated to 10M characters: about 108 minutes.
- **contextual** (live Azure chat, sample): 143 chunks in 32.4 s. Wrote context for 40 of 143 chunks with gpt-4o-mini. Only the first 40 chunks were enriched (max chunks = 40); set it to 0 to enrich all 143 (one chat call each).
- **proposition** (live Azure chat, sample): 160 chunks in 18.7 s. Rewrote 12 of 143 passages into 160 chunk(s) with gpt-4o-mini. Chunk text is the model's wording, so chunks are not highlighted in the document; each statement was kept only if at least 70% of its content words occur in its source passage (9 dropped). Only the first 12 passages were processed (max p

## 3. End to end through the API (upload, parse, embed with Azure `text-embedding-3-small`, search)

| Corpus | Files | Ingest s | Files/s | Parse failures | Chunks | Tokens embedded | Tokens billed | Cache hits | Index build s |
|---|---|---|---|---|---|---|---|---|---|
| 10k | 5 | 3.4 | 1.47 | 0 | 129 | 42,055 | 42,385 | 0 | 9.55 |

Retrieval on the same needle questions as the haystack grows (k = 5). `hit` = the right document was returned; `keyword coverage` = the retrieved text contained the distinctive keyword from the source passage. Confidence intervals are 95% bootstrap.

| Corpus | Mode | Questions | Hit | Hit 95% CI | MRR | nDCG | Keyword coverage | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| 10k | dense | 26 | 0.962 | 0.88 to 1.00 | 0.962 | 0.962 | 1.000 | 1,350 | 1,893 |
| 10k | hybrid | 26 | 0.962 | 0.88 to 1.00 | 0.933 | 0.940 | 0.962 | 54 | 191 |

## 4. Vector stores on the largest corpus's chunks

129 vectors, 1536 dimensions, 129 query vectors. Recall@10 is measured against FAISS exact search.

| Store | Load s | Search p50 ms | p95 ms | Recall@10 vs exact | Filtered p50 ms |
|---|---|---|---|---|---|
| faiss | 0.0 | 0.2 | 0.3 | 1.000 | 0.4 |
| qdrant | 2.0 | 4.6 | 5.6 | 1.000 | 10.8 |
| chroma | 0.3 | 10.3 | 13.3 | 1.000 | 12.8 |
| lancedb | 0.2 | 27.2 | 37.7 | 1.000 | 32.2 |
| pgvector | 0.6 | 56.4 | 61.3 | 1.000 | 57.1 |

## Reproduce

```
python scripts/scale_bench/bench_chunking.py 1000000 3000000 10000000
python scripts/scale_bench/bench_models.py
# start a second API with its own data folder:
#   STUDIO_DATA_DIR=data/scale/studio STUDIO_JOB_WORKERS=4 uvicorn studio.main:app --port 8001
python scripts/scale_bench/bench_pipeline.py http://localhost:8001 1000000 3000000 10000000
python scripts/scale_bench/bench_stores.py
python scripts/scale_bench/make_report.py
```
