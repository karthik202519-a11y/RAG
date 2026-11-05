"use client";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const TOKEN_KEY = "studio.token";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export function getToken(): string | null {
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}
export function setToken(t: string | null) {
  try {
    if (t) window.localStorage.setItem(TOKEN_KEY, t);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage can be blocked; the session then lasts until reload */
  }
}

function detailMessage(body: unknown, fallback: string): string {
  const d = (body as { detail?: unknown })?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => (x?.msg ? `${(x.loc ?? []).slice(1).join(".")}: ${x.msg}` : String(x))).join("; ");
  return fallback;
}

export async function api<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  let body = init.body;
  if (init.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(init.json);
  }
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { ...init, headers, body });
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_URL}. Is the backend running?`);
  }
  if (res.status === 204) return undefined as T;
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    if (res.status === 401 && token) {
      setToken(null);
      window.dispatchEvent(new Event("studio:logout"));
    }
    throw new ApiError(res.status, detailMessage(data, `${res.status} ${res.statusText}`));
  }
  return data as T;
}

/** Fetch a binary resource (e.g. a page image) with the auth header and return an object URL. */
export async function apiBlobUrl(path: string): Promise<string> {
  const token = getToken();
  const res = await fetch(`${API_URL}${path}`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText}`);
  return URL.createObjectURL(await res.blob());
}

// ---- types mirrored from the backend -------------------------------------------------
export type User = { id: string; email: string; display_name: string };
export type Workspace = { id: string; name: string; role: "owner" | "editor" | "viewer" };
export type Project = { id: string; workspace_id: string; name: string; description: string; created_at: string };
export type Collection = { id: string; name: string; description: string };

export type Version = {
  id: string; version: number; sha256: string; size_bytes: number; ext: string;
  status: "uploaded" | "processing" | "ready" | "failed"; error: string | null;
  pages: number | null; char_count: number | null; token_count: number | null; table_count: number | null;
  parser: string | null; extraction_seconds: number | null; created_at: string;
};
export type Doc = {
  id: string; project_id: string; collection_id: string | null; filename: string; tags: string[];
  doc_metadata: Record<string, unknown>; created_at: string; latest_version: Version | null;
  version_count: number; versions?: Version[];
};
export type Job = {
  id: string; kind: string; subject_id: string; status: "queued" | "running" | "succeeded" | "failed" | "cancelled" | "dead";
  attempts: number; max_attempts: number; progress: number; stage: string | null; error: string | null;
  filename: string | null; created_at: string; started_at: string | null; finished_at: string | null;
};
export type IngestionStats = {
  documents: number; versions: number; pages: number; characters: number; tokens: number; tables: number;
  storage_bytes: number; by_type: Record<string, number>; by_status: Record<string, number>;
  extraction_success_rate: number | null; throughput_pages_per_second: number | null; queue_depth: number;
  jobs: Record<string, number>; estimated_ingestion_cost_usd: number;
};

export type ExploreResponse = {
  dimension: number; sampled: number; total_chunks: number; explained_variance: number[]; method: string; warning: string;
  points: { chunk_id: string; x: number; y: number; filename: string; chunk_index: number; type: string; pages: number[]; preview: string }[];
};
export type SimilarityResponse = {
  query_norm: number; dimension: number; note: string;
  rows: { chunk_id: string; filename: string; chunk_index: number; preview: string; cosine: number; dot: number; euclidean: number; norm: number;
    rank_cosine: number; rank_dot: number; rank_euclidean: number }[];
};
export type EvalQuestion = { id: string; position: number; question: string; qtype: string; answerable: boolean;
  evidence: { document: string; pages: number[] }[]; keywords: string[]; reference_answer: string | null; split: string; approved: boolean };
export type EvalDatasetInfo = { id: string; name: string; description: string; questions: number; runs: number };
export type EvalDatasetDetail = { id: string; name: string; description: string; questions: EvalQuestion[]; splits: Record<string, number>;
  answerable: number; unanswerable: number; runs_on_test_split: number; guidance: string };
export type MetricAgg = { mean: number | null; n: number; ci95: [number, number] | null };
export type EvalRunInfo = {
  id: string; label: string; index_name: string; status: string; error: string | null; seconds: number | null; created_at: string;
  config: { mode: string; k: number; split: string; transform: string; rerank: string; mmr: boolean; with_answers: boolean; judge?: boolean };
  job: { progress: number; stage: string | null; status: string } | null;
  metrics: null | {
    questions: number; answerable: number; unanswerable: number; k: number; failed_questions?: number;
    retrieval: Record<string, MetricAgg>; by_type: Record<string, { n: number; hit: number | null; mrr: number | null; ndcg: number | null; recall: number | null }>;
    latency_ms: { p50: number | null; p95: number | null; mean: number | null }; llm_tokens: number; notes: string[];
    unanswerable_gate: { top1_dense_unanswerable: { min: number; max: number } | null; top1_dense_answerable: { min: number; median: number } | null;
      sweep: { threshold: number; answerable_kept: number; unanswerable_refused: number }[] | null };
    answers: null | { n: number; tokens: number; not_available: string[]; abstention_correct: { rate: number | null; n: number };
      citations_resolve: { rate: number | null; n: number }; keyword_hit: { rate: number | null; n: number };
      faithfulness?: { mean: number; n: number; contradicted_claims: number; judge_failures: number };
      relevance?: { yes: number; partial: number; no: number; n: number } };
  };
  per_question?: { question_id: string; question: string; qtype: string; answerable: boolean; split: string; error?: string;
    metrics: Record<string, number | null>; latency_ms: number; top1_dense_score: number | null;
    top: { rank: number; filename: string; pages: number[]; score: number }[] }[];
};
export type EvalCompare = {
  runs: { id: string; label: string; index: string }[];
  metrics: { metric: string; values: Partial<MetricAgg>[] }[];
  paired_vs_first: { other: string; shared_questions: number;
    [m: string]: unknown; changed_questions: { question_id: string; question: string; baseline_rank: number | null; other_rank: number | null; direction: string }[] }[];
  note: string;
};
export type FiguresInfo = {
  document_id: string; version_id: string; described: number; pending?: boolean;
  figures: { page: number; has_image: boolean; described: boolean; model: string | null; description: string | null }[];
  vision: { configured: boolean; detail: string };
};
export type Provider = { id: string; model: string | null; available: boolean; detail: string; semantic: boolean };
export type Providers = {
  embedding: Provider[]; vector_stores: { id: string; available: boolean; mode: string; detail?: string }[];
  generation: { configured: boolean; detail: string; default_model: string | null; models: string[] };
};
export type IndexInfo = {
  id: string; project_id: string; name: string; status: "building" | "ready" | "failed"; error: string | null;
  provider: string; model: string; dimension: number | null; metric: string; store: string; collection: string;
  chunk_count: number | null; tokens_embedded: number | null; tokens_billed: number | null; cache_hits: number | null;
  build_seconds: number | null; created_at: string; chunking: { method: string; params: Record<string, unknown> };
  documents: number; section_prefix: boolean; stored_vectors?: number; chunks_by_document?: Record<string, number>;
  job: { id: string; status: string; progress: number; stage: string | null; attempts: number; error: string | null } | null;
};
export type Contribution = {
  graph?: { rank: number; score: number }; tfidf?: { rank: number; score: number; matched_terms: string[] }; dense?: { rank: number; score: number }; bm25?: { rank: number; score: number; matched_terms: string[] }; rrf?: number;
};
export type SearchHit = {
  final_rank?: number; queries_found_by?: number;
  rerank?: { score: number; before_rank: number; after_rank: number; moved: number; method?: string };
  mmr?: { relevance: number; max_similarity_to_selected: number };
  rank: number; chunk_id: string; score: number; contributions: Contribution; document_id: string; version_id: string;
  filename: string; chunk_index: number; type: "text" | "table"; section: string | null; pages: number[];
  printed_pages: (string | null)[]; start: number | null; end: number | null; tokens: number; citation: string; text: string;
};
export type AnswerResponse = {
  question: string; model: string; mode: string; strict: boolean; answer: string; abstained: boolean;
  abstain_reason?: string; empty?: boolean; truncated?: boolean;
  prompt: { messages: { role: string; content: string }[]; system_tokens: number; context_tokens: number;
    question_tokens: number; context_budget: number } | null;
  context_passages: (SearchHit & { n: number })[];
  dropped_passages: { citation: string; rank: number }[];
  citations: { cited: number[]; unresolved: number[]; all_resolve: boolean; has_citations: boolean };
  usage: { prompt_tokens: number; completion_tokens: number; reasoning_tokens: number; served_by: string;
    finish_reason: string | null; estimated_cost_usd: number | null } | null;
  timings_ms: Record<string, number>;
};
export type PipelineStage = { stage: string; ms: number; method?: string; queries?: { text: string; role: string; search: boolean; mode?: string }[];
  llm_tokens?: number; n_queries?: number; per_query?: { query: string; role: string; mode: string; n: number }[]; unique_chunks?: number; pool?: number;
  warning?: string | null; lambda?: number; changed_top_k?: boolean; moves?: { chunk_id: string; score: number; before_rank: number; after_rank: number }[] };
export type SearchResponse = {
  stages?: PipelineStage[]; llm_tokens?: number; warnings?: string[];
  query: string; mode: string; top_k: number; candidates: number; results: SearchHit[]; run_id: string;
  timings_ms: Record<string, number>;
  info: { mode: string; query_terms?: string[]; query_embedding?: {
    provider: string; model: string; dimension: number; norm: number; tokens_billed: number; cached: boolean; first_values: number[] } };
};

export type ParamSpec = {
  name: string; label: string; type: "int" | "float" | "bool" | "select" | "text"; default: unknown;
  min: number | null; max: number | null; step: number | null; options: string[] | null; unit: string | null; help: string;
};
export type Method = {
  id: string; category: string; stage: string; name: string; summary: string; how_it_works: string;
  advantages: string[]; disadvantages: string[]; suitable_documents: string[]; use_cases: string[];
  params: ParamSpec[]; inputs: string[]; outputs: string[]; requires: string[]; performance_notes: string;
  cost_notes: string; needs_cloud_credentials: boolean; status: "implemented" | "requires_credentials" | "unavailable";
  unavailable_reason: string | null; references: string[]; limitations: string[]; compare_with: string[];
  recommended_experiment: string;
};
export type Catalog = { items: Method[]; categories: Record<string, number>; counts: Record<string, number> };

export type Chunk = {
  index: number; start: number | null; end: number | null; text: string | null; type: "text" | "table";
  chars: number; tokens: number; pages: number[]; printed_pages: (string | null)[]; section: string | null;
  overlap_prev_chars: number; ends_mid_sentence: boolean;
};
export type ChunkStats = {
  chunk_count: number; text_chunks: number; table_chunks: number; doc_chars: number; tokens_total: number;
  tokens_mean: number; tokens_median: number; tokens_p10: number; tokens_p90: number; tokens_min: number;
  tokens_max: number; histogram: { from: number; to: number; count: number }[]; duplication_ratio: number;
  coverage_ratio: number; mid_sentence_pct: number; estimated_embedding_tokens: number;
  estimated_embedding_cost_usd: number | null; vectors_to_store: number; processing_seconds: number;
};
export type ChunkResult = {
  method: string; label: string; params: Record<string, unknown>; chunks: Chunk[]; stats: ChunkStats; warnings: string[];
};
export type ChunkPreview = {
  document_id: string; version_id: string; filename: string; page_from: number; page_to: number;
  doc_text: string; results: ChunkResult[];
};
