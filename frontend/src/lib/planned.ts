/** Screens in the product spec that are not built yet. Shown honestly in the nav and on /planned/<slug>. */
export type Planned = {
  slug: string; title: string; stage: string; summary: string;
  willDo: string[]; backendToday: string;
};

export const PLANNED: Planned[] = [
  { slug: "parsing", title: "Parsing & Extraction Lab", stage: "Stage 5",
    summary: "Compare parsers side by side on one document: text, tables, figures, OCR regions.",
    willDo: ["Run PyMuPDF, pdfplumber, Docling, OCR and cloud parsers on the same file", "Highlight the PDF region behind each extracted block", "Show extraction time, cost and errors per parser"],
    backendToday: "One PDF parser (PyMuPDF) and six text-like parsers run in the ingestion job. Parser metadata is in the Method Catalog; there is no side-by-side comparison yet." },
  { slug: "multimodal", title: "Multimodal Lab", stage: "Stage 5",
    summary: "Index figures, charts and page images; query them with text.",
    willDo: ["Figure crops with vision-model descriptions", "OCR regions and captions", "Image and page-image embeddings"],
    backendToday: "Diagram pages are detected and rendered to PNG during PDF ingestion; nothing reads them yet." },
  { slug: "graph", title: "GraphRAG Explorer", stage: "Stage 5",
    summary: "Build a knowledge graph from your documents and compare graph retrieval with vector retrieval.",
    willDo: ["Entity and relationship extraction with provenance", "Communities and local/global search", "Interactive graph canvas"],
    backendToday: "Not started." },
  { slug: "generation", title: "Prompt & Generation Lab", stage: "Stage 2-3",
    summary: "Configure the LLM, inspect the exact prompt and verify citations.",
    willDo: ["Azure OpenAI chat deployment", "Prompt inspector with token counts", "Grounded answers with resolvable citations"],
    backendToday: "Not started; needs your Azure chat deployment name." },
  { slug: "pipelines", title: "Pipeline Builder", stage: "Stage 6",
    summary: "Compose a RAG pipeline visually and run it with artifact reuse.",
    willDo: ["Node-based builder with type checking", "Templates and versioning", "Live execution DAG"],
    backendToday: "Not started." },
  { slug: "experiments", title: "Experiment Studio", stage: "Stage 6",
    summary: "Run configurations against the same questions and compare quality, latency and cost.",
    willDo: ["Immutable configurations and isolated indexes", "Side-by-side metrics with example drill-down", "Exportable reports"],
    backendToday: "Not started. The chunking comparison in the Chunking Playground is the first building block." },
  { slug: "traces", title: "Traces & Monitoring", stage: "Stage 7",
    summary: "OpenTelemetry traces for every pipeline run.",
    willDo: ["Span waterfall per run", "Token and cost accounting", "Langfuse export"],
    backendToday: "Job status, attempts and timing are recorded per ingestion job; no tracing yet." },
  { slug: "settings", title: "Integrations & Settings", stage: "Stage 1-2",
    summary: "Provider connections, credentials and security settings.",
    willDo: ["Azure OpenAI / AI Search / Blob / Document Intelligence wizard", "Encrypted secret storage", "Connection health"],
    backendToday: "Providers are configured through environment variables (.env). No settings UI yet." },
];
