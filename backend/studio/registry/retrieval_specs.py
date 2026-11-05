"""Registry entries: embedding providers, vector stores, retrieval strategies."""
from .base import MethodSpec, ParamSpec, registry


def _na(id_, cat, stage, name, summary, why, creds=False):
    registry.register(MethodSpec(id=id_, category=cat, stage=stage, name=name, summary=summary,
                                 how_it_works="Not implemented yet.", advantages=[], disadvantages=[],
                                 status="unavailable", unavailable_reason=why, needs_cloud_credentials=creds))


# ---------------------------------------------------------------- embedding providers
registry.register(MethodSpec(
    id="azure_openai", category="embedding", stage="Embedding", name="Azure OpenAI embeddings",
    summary="Semantic embeddings from your Azure OpenAI / AI Foundry deployment (for example text-embedding-3-small).",
    how_it_works="Each chunk is sent to the embedding deployment, which returns one vector (for text-embedding-3-small, 1,536 "
                 "numbers). Vectors are L2-normalised so cosine similarity equals the dot product. Results are cached by "
                 "content hash, so identical text is billed once across indexes.",
    advantages=["Strong semantic matching", "Managed service, no GPU", "Cached: re-indexing unchanged text is free"],
    disadvantages=["Needs credentials and network", "Billed per token", "Input limited by the model's context"],
    needs_cloud_credentials=True, status="requires_credentials", requires=["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                                                                           "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    cost_notes="Billed per input token; the build reports tokens billed versus cache hits.",
    limitations=["One deployment is configured through environment variables; a provider settings UI is not built yet"],
    compare_with=["local_hash"],
    recommended_experiment="Build two indexes with different chunk sizes on the same documents and compare retrieval for your real questions."))
registry.register(MethodSpec(
    id="local_hash", category="embedding", stage="Embedding", name="Offline hashing embedder (demo only)",
    summary="A free, offline stand-in that counts shared words. It is NOT semantic.",
    how_it_works="Hashes each non-stopword into one of 256 buckets and counts them. Two texts are similar only when they share "
                 "words, so it behaves like a crude keyword match with a fixed-size vector.",
    advantages=["Works with no credentials", "Deterministic and instant", "Lets you try the whole pipeline offline"],
    disadvantages=["No understanding of meaning or synonyms", "Not suitable for judging real retrieval quality"],
    limitations=["Use it to learn the workflow and for tests, never to evaluate a configuration"],
    compare_with=["azure_openai"]))
for _id, _n, _why, _c in [
    ("openai", "OpenAI embeddings", "Planned (Stage 3); needs an OpenAI key.", True),
    ("sentence_transformers", "Sentence Transformers / BGE / E5 (local)", "Planned (Stage 3); needs model downloads and torch.", False),
    ("ollama", "Ollama local embeddings", "Planned (Stage 3); needs an Ollama server.", False),
    ("cohere", "Cohere embeddings", "Planned (Stage 3); needs a Cohere key.", True),
    ("voyage", "Voyage AI embeddings", "Planned (Stage 3); needs a Voyage key.", True),
    ("clip", "CLIP / SigLIP multimodal embeddings", "Planned (Stage 5).", False),
]:
    _na(_id, "embedding", "Embedding", _n, f"{_n}.", _why, _c)

# ------------------------------------------------------------------- vector stores
registry.register(MethodSpec(
    id="qdrant", category="vector_store", stage="Indexing", name="Qdrant",
    summary="Open-source vector database. Runs embedded (on disk, in-process) or as a server.",
    how_it_works="Each index gets its own collection with a fixed vector size and cosine distance. Points carry the chunk id and "
                 "filterable payload (document, chunk type). Search uses Qdrant's HNSW index with payload filters.",
    advantages=["Real vector database engine locally with no setup", "Payload filtering", "Server mode for scale"],
    disadvantages=["Embedded mode allows one process", "HNSW settings are not exposed in the UI yet"],
    limitations=["Distance is fixed to cosine for now", "Quantization and HNSW parameters are not configurable yet"],
    compare_with=["pgvector", "chroma"]))
registry.register(MethodSpec(
    id="faiss", category="vector_store", stage="Indexing", name="FAISS (exact, flat)",
    summary="Meta's open-source similarity-search library, used here as an exact brute-force index kept in memory and saved to disk.",
    how_it_works="Vectors are stored in a flat FAISS index (cosine = inner product on normalised vectors). Every query is compared with "
                 "every vector, so recall is 100% by construction. Metadata filters run inside the search through an id selector.",
    advantages=["Exact results: the baseline to judge approximate indexes against", "No server, tiny dependency", "Very fast up to a few hundred thousand vectors"],
    disadvantages=["Search time grows linearly with the number of vectors", "Whole index lives in memory", "One process: no concurrent writers or replication"],
    limitations=["Only the flat index is exposed; HNSW and IVF are not", "Filtering scans metadata in Python before searching"],
    compare_with=["qdrant"]))
for _id, _n, _why in [
    ("pgvector", "PostgreSQL + pgvector", "Planned (Stage 4); Postgres runs in compose but without the pgvector extension image."),
    ("chroma", "Chroma", "Planned (Stage 4)."),
    ("azure_ai_search", "Azure AI Search", "Planned (Stage 4); needs an Azure AI Search service."),
    ("pinecone", "Pinecone", "Planned (Stage 4); needs an API key."), ("milvus", "Milvus", "Planned (Stage 4)."),
    ("weaviate", "Weaviate", "Planned (Stage 4)."), ("elasticsearch", "Elasticsearch / OpenSearch", "Planned (Stage 4)."),
    ("lancedb", "LanceDB", "Planned (Stage 4)."), ("mongodb_atlas", "MongoDB Atlas Vector Search", "Planned (Stage 4)."),
]:
    _na(_id, "vector_store", "Indexing", _n, f"{_n}.", _why)

# ----------------------------------------------------------------------- retrieval
_COMMON = [
    ParamSpec(name="top_k", label="Results (top-k)", type="int", default=5, min=1, max=50, help="How many chunks to return."),
    ParamSpec(name="score_threshold", label="Minimum score", type="float", default=None, help="Drop results scoring below this."),
]
registry.register(MethodSpec(
    id="dense", category="retrieval", stage="Retrieval", name="Dense vector search",
    summary="Embed the question and return the chunks whose vectors are closest.",
    how_it_works="The query is embedded with the index's own model (dimension is checked), then Qdrant returns the nearest "
                 "chunks by cosine similarity.",
    advantages=["Matches meaning, not just words", "Handles paraphrases"],
    disadvantages=["Can miss exact identifiers and rare terms", "Score scale depends on the model"],
    params=_COMMON, compare_with=["bm25", "hybrid"],
    recommended_experiment="Search an exact code or acronym in dense and BM25 mode and compare the ranks."))
registry.register(MethodSpec(
    id="bm25", category="retrieval", stage="Retrieval", name="BM25 keyword search",
    summary="Classic keyword ranking: chunks sharing rare query words score highest.",
    how_it_works="Tokenizes chunks and the query into lowercase words, then scores with BM25 (term frequency, rarity across "
                 "chunks, length normalisation). Results report which query terms matched.",
    advantages=["Exact terms, codes and acronyms", "No embeddings, free and instant"],
    disadvantages=["No synonyms or paraphrase understanding", "Sensitive to wording"],
    params=_COMMON, compare_with=["dense", "hybrid"]))
registry.register(MethodSpec(
    id="hybrid", category="retrieval", stage="Retrieval", name="Hybrid (dense + BM25, reciprocal rank fusion)",
    summary="Run both searches and merge the rankings.",
    how_it_works="Each retriever returns candidates; a chunk's fused score is the sum of 1 / (k + rank) over the retrievers that "
                 "found it. Each result shows its dense and BM25 rank and score so you can see who contributed.",
    advantages=["Covers meaning and exact terms", "No score normalisation needed"],
    disadvantages=["Two searches per query", "The constant k and candidate count are knobs to tune on your data"],
    params=_COMMON + [
        ParamSpec(name="candidates", label="Candidates per retriever", type="int", default=20, min=1, max=200,
                  help="How many results each retriever contributes before fusion."),
        ParamSpec(name="rrf_k", label="RRF constant k", type="int", default=60, min=1, max=1000,
                  help="Larger k flattens the advantage of top ranks.")],
    compare_with=["dense", "bm25"]))
# ---------------------------------------------------------------------- generation
registry.register(MethodSpec(
    id="azure_openai_chat", category="generation", stage="Generation", name="Grounded answer (Azure OpenAI chat)",
    summary="Retrieve passages, then ask a chat model to answer only from them, with numbered citations you can check.",
    how_it_works="The top passages are numbered and packed into the prompt within a token budget (lower-ranked ones are dropped, "
                 "and the UI says which). In strict mode the system prompt allows only the supplied passages and demands a "
                 "fixed refusal sentence when they do not contain the answer. After generation every [n] marker is checked "
                 "against the passages actually sent. An optional minimum score skips the model call for weak retrievals.",
    advantages=["Citations are verified against what was sent", "The exact prompt is inspectable", "Abstains instead of guessing"],
    disadvantages=["Costs tokens per question", "A model can still misread a passage while citing it correctly",
                   "Citation checking proves the number exists, not that the claim is supported"],
    needs_cloud_credentials=True, status="requires_credentials",
    requires=["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_DEPLOYMENT"],
    params=[ParamSpec(name="strict", label="Grounded mode", type="bool", default=True,
                      help="Only the supplied passages may be used; otherwise the model must refuse."),
            ParamSpec(name="context_budget", label="Context budget", type="int", default=3000, min=20, max=30000, unit="tokens",
                      help="Maximum tokens of retrieved text in the prompt."),
            ParamSpec(name="max_output_tokens", label="Max output tokens", type="int", default=600, min=16, max=8000, unit="tokens",
                      help="Reasoning models (gpt-5 family) spend part of this on thinking, so allow more."),
            ParamSpec(name="min_score", label="Abstain below score", type="float", default=None,
                      help="If the best dense cosine score is lower, refuse without calling the model.")],
    limitations=["Streaming is not implemented", "Answers are not persisted yet", "Faithfulness is not scored; that needs the Evaluation Center"],
    recommended_experiment="Ask an on-topic and an off-topic question in grounded mode and read the prompt inspector for each."))
for _id, _n, _why in [
    ("openai_chat", "OpenAI chat", "Planned (Stage 3)."), ("anthropic", "Anthropic Claude", "Planned (Stage 3); needs an API key."),
    ("bedrock", "AWS Bedrock", "Planned (Stage 3)."), ("gemini", "Google Gemini", "Planned (Stage 3)."),
    ("ollama_chat", "Ollama local models", "Planned (Stage 3)."),
]:
    _na(_id, "generation", "Generation", _n, f"{_n}.", _why, True)

for _id, _n, _why in [
    ("mmr", "Maximum marginal relevance", "Planned (Stage 4)."), ("multi_query", "Multi-query retrieval", "Planned (Stage 4); needs an LLM."),
    ("hyde", "HyDE", "Planned (Stage 4); needs an LLM."), ("query_rewrite", "Query rewriting", "Planned (Stage 4); needs an LLM."),
    ("parent_document", "Parent-document retrieval", "Planned (Stage 4); needs parent-child chunking."),
    ("rerank_cross_encoder", "Cross-encoder reranking", "Planned (Stage 4)."),
    ("graph_retrieval", "Graph-assisted retrieval", "Planned (Stage 5)."),
]:
    _na(_id, "retrieval", "Retrieval", _n, f"{_n}.", _why)


# ------------------------------------------------------------------------ multimodal
registry.register(MethodSpec(
    id="vision_describe_page", category="multimodal", stage="Visual content", name="Vision description of figure pages",
    summary="A vision model reads each diagram or chart page and writes a description that becomes searchable text.",
    how_it_works="Pages flagged at ingestion (many vector drawings) are rendered to PNG. Each image is sent to a vision-capable "
                 "deployment with a prompt asking for the title, every label, relationships between elements, axes, units, and "
                 "readable values. The description is stored in the page as a text item starting '[Figure description, page N]', so "
                 "every chunker and index can use it. Progress is saved per page, so a cancel or crash repeats nothing paid.",
    advantages=["Makes diagrams and charts searchable by their content", "Works with any chunker", "Checkpointed and retryable"],
    disadvantages=["Costs vision tokens per page", "The model can misread dense charts or invent structure; descriptions are not verified",
                   "Works on whole pages, not on cropped figures"],
    needs_cloud_credentials=True, status="requires_credentials",
    requires=["AZURE_OPENAI_VISION_DEPLOYMENT (a vision-capable deployment such as gpt-4o)"],
    params=[ParamSpec(name="pages", label="Pages", type="text", default=None, help="Only these flagged pages; empty = all."),
            ParamSpec(name="force", label="Describe again", type="bool", default=False, help="Replace existing descriptions.")],
    cost_notes="Roughly 1k input tokens per page plus the description; a 29-page handbook subset costs a few cents to tens of cents.",
    limitations=["Page-level, not figure-level: descriptions mix every figure on the page", "Values read from charts are approximate",
                 "Page detection uses drawing counts and misses figures built from embedded images"],
    compare_with=["ocr_tesseract", "clip_embeddings"],
    recommended_experiment="Describe the flagged pages, index with and without them, and ask a question whose answer is only in a diagram."))
for _id, _n, _why in [
    ("ocr_tesseract", "OCR (Tesseract / PaddleOCR)", "Planned (Stage 5); scanned PDFs are unsupported until OCR exists."),
    ("figure_crop", "Figure-region crop + description", "Planned (Stage 5); needs figure detection."),
    ("clip_embeddings", "CLIP / SigLIP image embeddings", "Planned (Stage 5)."),
    ("page_image_retrieval", "Page-screenshot retrieval (ColPali-style)", "Planned (Stage 5)."),
    ("chart_to_table", "Chart to data table", "Planned (Stage 5)."),
]:
    _na(_id, "multimodal", "Visual content", _n, f"{_n}.", _why)
