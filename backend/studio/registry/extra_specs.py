"""Registry entries that replace earlier 'planned' placeholders now that the methods exist, and state precisely why the rest cannot run.

Status vocabulary: implemented = runs here and is tested; requires_credentials = implemented but needs a key or server that the
repository does not ship (adapters for hosted services are tested against recorded responses only, which each entry says);
unavailable = not built, with the exact reason.
"""
from .base import MethodSpec, ParamSpec, registry


def put(**kw) -> None:
    kw.setdefault("advantages", [])
    kw.setdefault("disadvantages", [])
    registry.register(MethodSpec(**kw), replace=True)


NOT_TESTED_LIVE = "The adapter is tested against recorded responses in the service's documented format; it has not been run against the live service."

# ------------------------------------------------------------------------------ embeddings
put(id="fastembed", category="embedding", stage="Embedding", name="FastEmbed local models (BGE, E5, MiniLM ...)",
    summary="Real semantic embeddings computed on your own machine with ONNX models: no key, no server, no GPU.",
    how_it_works="Loads a sentence-embedding model (default BAAI/bge-small-en-v1.5, 384 dimensions) through ONNX Runtime and embeds each chunk locally. "
                 "Vectors are L2-normalised. The model downloads once (about 70 MB for the default) and then works offline. Change FASTEMBED_MODEL to use another supported model.",
    advantages=["Free and private: text never leaves the machine", "Semantic, unlike the hashing demo", "Cached by content hash like every provider"],
    disadvantages=["Slower than a hosted API on a CPU", "Smaller models trail large hosted models on hard queries"],
    requires=["fastembed", "ONNX Runtime"], cost_notes="No per-token cost; uses CPU time.", compare_with=["azure_openai", "local_hash"],
    recommended_experiment="Build one index with this and one with Azure OpenAI embeddings and compare them in the evaluation grid.")
put(id="sentence_transformers", category="embedding", stage="Embedding", name="Sentence Transformers (PyTorch)",
    summary="The PyTorch library behind many open embedding models. Covered here by FastEmbed.", how_it_works="Not offered separately.",
    status="unavailable", unavailable_reason="Not installed: it needs PyTorch (about 2 GB). The same BGE and E5 model families run through FastEmbed (ONNX) without it.",
    compare_with=["fastembed"])
put(id="openai", category="embedding", stage="Embedding", name="OpenAI embeddings",
    summary="Embeddings from the OpenAI API (default text-embedding-3-small).",
    how_it_works="Sends batches of chunks to /v1/embeddings and L2-normalises the vectors. Needs OPENAI_API_KEY.",
    advantages=["Strong general-purpose model"], disadvantages=["Billed per token", "Text leaves your machine"],
    status="requires_credentials", needs_cloud_credentials=True, requires=["OPENAI_API_KEY"], limitations=[NOT_TESTED_LIVE], compare_with=["azure_openai"])
put(id="ollama", category="embedding", stage="Embedding", name="Ollama local embeddings",
    summary="Embeddings from a model served by a local Ollama server (default nomic-embed-text).",
    how_it_works="Posts batches to {OLLAMA_URL}/api/embed. The provider shows as available only when the server answers and the model is pulled.",
    advantages=["Free and private", "Many models"], disadvantages=["Needs Ollama installed and the model pulled"],
    status="requires_credentials", requires=["Ollama server", "ollama pull nomic-embed-text"], limitations=[NOT_TESTED_LIVE + " Ollama is not installed on the test machine."])
put(id="cohere", category="embedding", stage="Embedding", name="Cohere embeddings",
    summary="Embeddings from the Cohere API (default embed-english-v3.0).", how_it_works="Posts to /v2/embed with input_type search_document.",
    advantages=["Strong retrieval models"], disadvantages=["Billed per token", "Queries are embedded as documents here, which Cohere advises against"],
    status="requires_credentials", needs_cloud_credentials=True, requires=["COHERE_API_KEY"], limitations=[NOT_TESTED_LIVE])
put(id="voyage", category="embedding", stage="Embedding", name="Voyage AI embeddings",
    summary="Embeddings from the Voyage AI API (default voyage-3).", how_it_works="Posts to /v1/embeddings with input_type document.",
    advantages=["Strong retrieval models"], disadvantages=["Billed per token"],
    status="requires_credentials", needs_cloud_credentials=True, requires=["VOYAGE_API_KEY"], limitations=[NOT_TESTED_LIVE])
put(id="clip", category="embedding", stage="Embedding", name="CLIP image and text embeddings (local)",
    summary="One embedding space for pictures and text, so a sentence can find a page by how it looks.",
    how_it_works="CLIP ViT-B/32 via FastEmbed. Page images and text queries are embedded into the same 512-dimension space; similarity ranks pages. "
                 "Used by page-image search in the document tools, not by the text index builder.",
    advantages=["Finds visual content that has no text", "Local, no key"], disadvantages=["Large first download (about 350 MB)", "Weak at reading small text in an image"],
    requires=["fastembed", "Pillow"], compare_with=["vision_describe_page"])

# ------------------------------------------------------------------------------ vector stores
put(id="chroma", category="vector_store", stage="Indexing", name="Chroma (embedded)",
    summary="Open-source embedded vector database with an HNSW index, stored on disk.",
    how_it_works="Each index is a Chroma collection with the chosen distance (cosine, inner product or L2). Metadata filters run inside the query.",
    advantages=["No server", "HNSW approximate search", "Simple"], disadvantages=["Single process", "Fewer tuning knobs than Qdrant"],
    compare_with=["qdrant", "faiss", "lancedb"])
put(id="lancedb", category="vector_store", stage="Indexing", name="LanceDB (embedded)",
    summary="Open-source embedded vector database storing columnar Lance files on disk.",
    how_it_works="Vectors and metadata live in a Lance table. Searches here are exact scans with the filter applied first, so recall is 100%; "
                 "an ANN index would be needed for very large tables.",
    advantages=["No server", "Exact results", "Columnar storage"], disadvantages=["No ANN index is built, so latency grows with size"],
    compare_with=["faiss", "chroma"])
put(id="pgvector", category="vector_store", stage="Indexing", name="PostgreSQL + pgvector",
    summary="Vectors in a PostgreSQL table with an HNSW index, next to your relational data.",
    how_it_works="A table per index with a vector column and an HNSW index (up to 2,000 dimensions; larger vectors are scanned exactly). "
                 "Queries use the cosine, inner-product or L2 operator and SQL filters. Needs PGVECTOR_URL.",
    advantages=["One database for everything", "SQL filters and transactions"], disadvantages=["Needs the pgvector extension (the pgvector/pgvector Docker image)"],
    status="requires_credentials", requires=["PGVECTOR_URL", "pgvector/pgvector:pg16"], compare_with=["qdrant"])
for _id, _name, _why in [
    ("milvus", "Milvus", "Needs a Milvus server (etcd, MinIO and Milvus containers); Milvus Lite has no Windows build. No adapter written."),
    ("weaviate", "Weaviate", "Needs a Weaviate server or its embedded binary download. No adapter written."),
    ("elasticsearch", "Elasticsearch / OpenSearch", "Needs a server (a large container image). No adapter written."),
    ("azure_ai_search", "Azure AI Search", "Needs an Azure AI Search service, which this project does not have. No adapter written."),
    ("pinecone", "Pinecone", "Needs a Pinecone account and key. No adapter written."),
    ("mongodb_atlas", "MongoDB Atlas Vector Search", "Needs an Atlas cluster. No adapter written."),
]:
    put(id=_id, category="vector_store", stage="Indexing", name=_name, summary=f"{_name} as a vector store.", how_it_works="Not implemented.",
        status="unavailable", unavailable_reason=_why)

# ------------------------------------------------------------------------------ retrieval
RET = dict(category="retrieval", stage="Retrieval")
put(id="mmr", name="Maximal marginal relevance (MMR)", summary="Re-pick results so they are relevant to the question but different from each other.",
    how_it_works="From the candidate pool, repeatedly choose the passage that maximises lambda x relevance minus (1 - lambda) x similarity to the passages already chosen.",
    advantages=["Removes near-duplicate hits", "No model calls"], disadvantages=["Can push a relevant duplicate out of the top k"], **RET)
put(id="multi_query", name="Multi-query retrieval", summary="A chat model writes several rewordings of the question; results are merged.",
    how_it_works="The model proposes alternative phrasings; each is searched; the rankings are fused with reciprocal rank fusion.",
    advantages=["Finds passages that use different words"], disadvantages=["One chat call per question", "More searches"],
    status="requires_credentials", needs_cloud_credentials=True, requires=["chat model"], **RET)
put(id="hyde", name="HyDE (hypothetical answer)", summary="A chat model writes a short imagined answer, and that is searched instead of the question.",
    how_it_works="The imagined answer is embedded, which often lands closer to real passages than a short question does; the original question is kept for keyword matching.",
    advantages=["Helps with short, vague questions"], disadvantages=["The imagined answer can mislead", "One chat call per question"],
    status="requires_credentials", needs_cloud_credentials=True, requires=["chat model"], **RET)
put(id="query_rewrite", name="Query rewriting", summary="A chat model rewrites the question into a clearer search query.",
    how_it_works="The rewritten query replaces the original for searching.", advantages=["Fixes typos and chatty phrasing"],
    disadvantages=["One chat call per question", "Can drop a detail"], status="requires_credentials", needs_cloud_credentials=True, requires=["chat model"], **RET)
put(id="decompose", name="Question decomposition", summary="A chat model splits a multi-part question into sub-questions that are searched separately.",
    how_it_works="Each sub-question is retrieved; results are fused.", advantages=["Helps with questions that need two facts"],
    disadvantages=["One chat call per question"], status="requires_credentials", needs_cloud_credentials=True, requires=["chat model"], **RET)
put(id="parent_document", name="Parent-document (small-to-big) retrieval", summary="Search small chunks, return the larger passage they came from.",
    how_it_works="Build an index with the Parent-child chunker, then choose 'expand to parent' (or neighbouring chunks) when searching or answering.",
    advantages=["Precise matching with fuller context"], disadvantages=["Needs a parent-child index", "More context tokens per hit"], **RET)
put(id="rerank_llm", name="LLM reranking", summary="A chat model scores each candidate 0 to 10 and the list is reordered.",
    how_it_works="The top candidates are sent in one prompt; the model returns a score per passage.", advantages=["Strong on subtle relevance"],
    disadvantages=["Costs tokens per question", "Slower"], status="requires_credentials", needs_cloud_credentials=True, requires=["chat model"], **RET)
put(id="rerank_cross_encoder", name="Cross-encoder reranking (local)", summary="A small local model reads the question and each passage together and reorders the candidates.",
    how_it_works="A cross-encoder (default Xenova/ms-marco-MiniLM-L-6-v2, via FastEmbed ONNX) scores every candidate; unlike a bi-encoder it sees both texts at once.",
    advantages=["Free, private, no tokens", "Usually more accurate than the first-stage ranking"], disadvantages=["Adds CPU time per query", "English-trained by default"],
    requires=["fastembed"], compare_with=["rerank_llm"], recommended_experiment="Run the evaluation grid with rerank none versus cross_encoder on the same index.", **RET)
put(id="graph_retrieval", name="Graph retrieval (GraphRAG local search)", summary="Start from entities named in the question and follow relations to the chunks they came from.",
    how_it_works="Entities and relations are extracted per chunk by a chat model and verified against the chunk text; a question's entities seed a walk of 1 or 2 hops.",
    advantages=["Follows explicit relationships", "Explains why a chunk was returned"], disadvantages=["Needs a graph build (one chat call per chunk)", "Only finds entities it extracted"],
    status="requires_credentials", needs_cloud_credentials=True, requires=["chat model (to build)"], **RET)

# ------------------------------------------------------------------------------ generation
GEN = dict(category="generation", stage="Generation", needs_cloud_credentials=True, status="requires_credentials")
put(id="openai_chat", name="OpenAI chat", summary="Answer with an OpenAI model: choose a model named openai:<model>.", how_it_works="Chat completions API.",
    requires=["OPENAI_API_KEY"], limitations=[NOT_TESTED_LIVE],
    advantages=["Strong general models", "Same prompt and citation checks as Azure"], disadvantages=["Billed per token", "Text leaves your machine"], **GEN)
put(id="anthropic", name="Anthropic Claude", summary="Answer with a Claude model: choose a model named anthropic:<model>.", how_it_works="Messages API.",
    requires=["ANTHROPIC_API_KEY"], limitations=[NOT_TESTED_LIVE],
    advantages=["Strong at following a grounding instruction", "Long context"], disadvantages=["Billed per token", "Text leaves your machine"], **GEN)
put(id="gemini", name="Google Gemini", summary="Answer with a Gemini model: choose a model named gemini:<model>.", how_it_works="generateContent API.",
    requires=["GEMINI_API_KEY"], limitations=[NOT_TESTED_LIVE],
    advantages=["Very long context", "Low cost tiers"], disadvantages=["Billed per token", "Text leaves your machine"], **GEN)
put(id="ollama_chat", name="Ollama local models", summary="Answer with a model served by a local Ollama server: choose ollama:<model>.", how_it_works="/api/chat.",
    requires=["Ollama server"], limitations=[NOT_TESTED_LIVE + " Ollama is not installed on the test machine."],
    advantages=["Free and private", "Works offline"], disadvantages=["Needs a capable machine", "Smaller local models follow citation rules less reliably"], **{**GEN, "needs_cloud_credentials": False})
put(id="bedrock", category="generation", stage="Generation", name="AWS Bedrock", summary="Models hosted on AWS Bedrock.", how_it_works="Not implemented.",
    status="unavailable", unavailable_reason="Needs AWS request signing (boto3) and an AWS account with model access. No adapter written.")

# ------------------------------------------------------------------------------ multimodal
MM = dict(category="multimodal", stage="Visual content")
put(id="ocr_tesseract", name="OCR (Tesseract)", summary="Tesseract OCR.", how_it_works="Not implemented.", status="unavailable",
    unavailable_reason="The Tesseract program is not installed. Local OCR is provided by RapidOCR (ONNX), which needs nothing installed.", **MM)
put(id="ocr_rapidocr", name="OCR (RapidOCR, local)", summary="Reads scanned PDFs and images with a local ONNX OCR model.",
    how_it_works="Each page is rendered at 200 dpi, text lines are detected and recognised, and lines are joined into paragraphs by vertical gaps. Choose the RapidOCR parser when uploading a scanned PDF, PNG or JPEG.",
    advantages=["No key, no server", "Works on scanned pages that have no text layer"], disadvantages=["Slower than reading a text layer (seconds per page)", "Tables come out as lines of text"],
    requires=["rapidocr", "onnxruntime"], compare_with=["vision_describe_page"], **MM)
put(id="figure_crop", name="Figure-region crops", summary="Finds the pictures and drawings on a page and cuts each out as its own image.",
    how_it_works="Embedded pictures and clusters of vector drawings are located with PyMuPDF, merged where they touch, filtered by size, and rendered at 150 dpi.",
    advantages=["Per-figure images instead of whole pages", "Local"], disadvantages=["Heuristic: very large or very small figures can be missed or merged"],
    requires=["PyMuPDF"], **MM)
put(id="clip_embeddings", name="CLIP image embeddings", summary="Embed page images and text into one space.", how_it_works="See the CLIP embedding entry.",
    requires=["fastembed"], advantages=["Local, no key"], disadvantages=["Large first download", "512-dimension space is coarse for fine detail"], **MM)
put(id="page_image_retrieval", name="Page-image retrieval (CLIP)", summary="Find pages by what they look like, with a text query.",
    how_it_works="Every page is rendered and embedded with CLIP; a query is embedded with the paired text model; pages are ranked by similarity.",
    advantages=["Works without any text on the page"], disadvantages=["Coarse: it matches layout and large text, not small print", "Not ColPali-style late interaction"],
    requires=["fastembed"], **MM)
put(id="chart_to_table", name="Chart to data table", summary="A vision model reads a chart or table image into a Markdown table.",
    how_it_works="The chosen figure (or whole page) is sent to the vision deployment with a prompt that asks for one table and '?' for unreadable values.",
    advantages=["Turns a chart into searchable numbers"], disadvantages=["Values are read by a model and can be wrong", "Costs vision tokens"],
    status="requires_credentials", needs_cloud_credentials=True, requires=["vision deployment"], **MM)
registry.register(registry.get("vision_describe_page").model_copy(update={"compare_with": ["ocr_rapidocr", "clip_embeddings", "chart_to_table"]}), replace=True)

# ------------------------------------------------------------------------------ parsers
PA = dict(category="parser", stage="Extraction")
put(id="pdfplumber", name="pdfplumber", summary="Reads digital PDFs with pdfminer; strong at ruled tables. Upload option and Parser Lab.",
    how_it_works="Text is extracted with the words inside detected tables removed, and each table becomes a Markdown table from its ruling lines.",
    advantages=["Finds ruled tables PyMuPDF missed on the NASA handbook"], disadvantages=["Slower", "No printed-page numbers or sections", "No diagram detection"],
    inputs=["pdf"], requires=["pdfplumber"], compare_with=["pymupdf"], **PA)
put(id="rapidocr", name="RapidOCR (local OCR)", summary="OCR for scanned PDFs and images. Upload option.",
    how_it_works="See OCR (RapidOCR, local).", advantages=["Reads pages with no text layer"], disadvantages=["Slow", "No table structure"],
    inputs=["pdf", "png", "jpg", "jpeg"], requires=["rapidocr"], compare_with=["pymupdf"], **PA)
put(id="azure_document_intelligence", name="Azure AI Document Intelligence", summary="Microsoft's layout model: text, tables and reading order from a PDF. Upload option.",
    how_it_works="Posts the file to prebuilt-layout, polls the operation, and maps lines and tables to pages.",
    advantages=["Strong tables and scans"], disadvantages=["Billed per page", "Needs a Document Intelligence resource"],
    status="requires_credentials", needs_cloud_credentials=True, inputs=["pdf"],
    requires=["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "AZURE_DOCUMENT_INTELLIGENCE_KEY"], limitations=[NOT_TESTED_LIVE], **PA)
for _id, _name, _why in [
    ("docling", "Docling", "Needs PyTorch and layout models (several GB). Install docling to use it; no adapter written."),
    ("unstructured", "Unstructured", "Pulls a large dependency tree (OCR, layout models). No adapter written."),
    ("tika", "Apache Tika", "Needs Java and a Tika server; Java is not installed. No adapter written."),
    ("tesseract", "Tesseract OCR", "The Tesseract program is not installed. RapidOCR covers local OCR."),
    ("paddleocr", "PaddleOCR", "Needs PaddlePaddle, which has no Python 3.14 build. RapidOCR runs the same PaddleOCR models through ONNX."),
    ("aws_textract", "AWS Textract", "Needs an AWS account and request signing. No adapter written."),
    ("google_document_ai", "Google Document AI", "Needs a Google Cloud project and credentials. No adapter written."),
    ("llamaparse", "LlamaParse", "Needs a LlamaCloud API key. No adapter written."),
]:
    put(id=_id, name=_name, summary=f"{_name} parser.", how_it_works="Not implemented.", status="unavailable", unavailable_reason=_why, **PA)
