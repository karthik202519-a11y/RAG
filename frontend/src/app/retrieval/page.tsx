"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { GitCompare, MessageSquareText, Search } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { AnswerPanel } from "@/components/answer-panel";
import { Badge, Button, Card, Empty, ErrorBox, Field, Input, PageHeader, Select, Spinner } from "@/components/ui";
import { SourcePanel, type SourceRef } from "@/components/source-panel";
import { api, type AnswerResponse, type Doc, type IndexInfo, type Providers, type SearchHit, type SearchResponse } from "@/lib/api";
import { useProject } from "@/lib/project";

type Mode = "dense" | "bm25" | "hybrid";
const MODES: { id: Mode; label: string; blurb: string }[] = [
  { id: "dense", label: "Dense (vector)", blurb: "Matches meaning. Embeds the question and finds the nearest chunk vectors." },
  { id: "bm25", label: "BM25 (keyword)", blurb: "Matches words. Best for exact terms, codes and acronyms." },
  { id: "hybrid", label: "Hybrid (RRF)", blurb: "Runs both and merges ranks: score = Σ 1/(k + rank)." },
];

function Contributions({ h, mode }: { h: SearchHit; mode: string }) {
  const c = h.contributions;
  return (
    <div className="flex flex-wrap gap-1.5 text-[11px]">
      {h.rerank && <Badge tone={h.rerank.moved > 0 ? "good" : h.rerank.moved < 0 ? "bad" : "neutral"} title="Rank before and after the LLM reranker, and its 0-10 score">
        rerank {h.rerank.before_rank}→{h.rerank.after_rank} · LLM {h.rerank.score}/10</Badge>}
      {h.mmr && <Badge title="Relevance to the query, and similarity to the already-selected results">mmr rel {h.mmr.relevance} · overlap {h.mmr.max_similarity_to_selected}</Badge>}
      {h.queries_found_by != null && h.queries_found_by > 1 && <Badge tone="accent">found by {h.queries_found_by} queries</Badge>}
      {c.dense && <Badge tone="accent" title="Rank and cosine similarity from the vector search">dense #{c.dense.rank} · cos {c.dense.score.toFixed(3)}</Badge>}
      {c.bm25 && <Badge tone="warn" title="Rank and BM25 score from the keyword search">bm25 #{c.bm25.rank} · {c.bm25.score.toFixed(2)}</Badge>}
      {c.bm25 && c.bm25.matched_terms.length > 0 && <Badge title="Query terms found in this chunk">matched: {c.bm25.matched_terms.join(", ")}</Badge>}
      {mode === "hybrid" && c.rrf != null && <Badge tone="good" title="Reciprocal rank fusion score">rrf {c.rrf.toFixed(4)}</Badge>}
    </div>
  );
}

export default function RetrievalLab() {
  const { project, loading } = useProject();
  const pid = project?.id;
  const indexes = useQuery({ queryKey: ["indexes", pid], enabled: !!pid, queryFn: () => api<IndexInfo[]>(`/projects/${pid}/indexes`) });
  const docs = useQuery({ queryKey: ["docs-ready", pid], enabled: !!pid, queryFn: () => api<{ items: Doc[] }>(`/projects/${pid}/documents?status=ready&limit=500`) });
  const ready = (indexes.data ?? []).filter((i) => i.status === "ready");

  const [indexState, setIndexId] = useState<string | null>(null);
  const wanted = typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("index");
  const indexId = indexState ?? ready.find((i) => i.id === wanted)?.id ?? ready[0]?.id ?? "";

  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("dense");
  const [topK, setTopK] = useState(5);
  const [threshold, setThreshold] = useState("");
  const [candidates, setCandidates] = useState(20);
  const [rrfK, setRrfK] = useState(60);
  const [docFilter, setDocFilter] = useState("");
  const [typeFilter, setTypeFilter] = useState("");
  const [picked, setPicked] = useState<SourceRef | null>(null);
  const providers = useQuery({ queryKey: ["providers"], queryFn: () => api<Providers>("/providers") });
  const gen = providers.data?.generation;
  const [modelState, setModel] = useState("");
  const model = modelState || gen?.default_model || "";
  const [strict, setStrict] = useState(true);
  const [maxOut, setMaxOut] = useState(600);
  const [budget, setBudget] = useState(3000);
  const [abstainBelow, setAbstainBelow] = useState("");
  const [transform, setTransform] = useState("none");
  const [rerank, setRerank] = useState("none");
  const [mmrOn, setMmrOn] = useState(false);
  const [mmrLambda, setMmrLambda] = useState(0.6);
  const advanced = transform !== "none" || rerank !== "none" || mmrOn;

  const body = (m: Mode) => ({
    query, mode: m, top_k: topK, candidates: m === "hybrid" ? candidates : null, rrf_k: rrfK,
    score_threshold: threshold === "" ? null : Number(threshold),
    document_ids: docFilter ? [docFilter] : null, types: typeFilter ? [typeFilter] : null,
  });
  const run = useMutation({
    mutationFn: () => advanced
      ? api<SearchResponse>(`/indexes/${indexId}/retrieve`, { method: "POST", json: {
          query, mode, top_k: topK, transform, rerank, mmr: mmrOn, mmr_lambda: mmrLambda, rrf_k: rrfK,
          llm_model: gen?.default_model ?? null, rerank_candidates: Math.max(topK, 15),
          score_threshold: threshold === "" ? null : Number(threshold),
          document_ids: docFilter ? [docFilter] : null, types: typeFilter ? [typeFilter] : null } })
      : api<SearchResponse>(`/indexes/${indexId}/search`, { method: "POST", json: body(mode) }),
    onSuccess: () => setPicked(null),
  });
  const ask = useMutation({
    mutationFn: () => api<AnswerResponse>(`/indexes/${indexId}/answer`, {
      method: "POST",
      json: { question: query, model, mode, top_k: topK, strict, max_output_tokens: maxOut, context_budget: budget,
        min_score: abstainBelow === "" ? null : Number(abstainBelow), document_ids: docFilter ? [docFilter] : null },
    }),
    onSuccess: () => setPicked(null),
  });
  const compare = useMutation({
    mutationFn: async () => {
      const out: Record<Mode, SearchResponse> = {} as Record<Mode, SearchResponse>;
      for (const m of ["dense", "bm25", "hybrid"] as Mode[]) out[m] = await api<SearchResponse>(`/indexes/${indexId}/search`, { method: "POST", json: body(m) });
      return out;
    },
  });

  if (loading || indexes.isLoading) return <Spinner />;
  if (!project) return <Empty title="No project selected"><Link className="text-accent underline" href="/projects">Create a project first</Link></Empty>;
  if (ready.length === 0) {
    return <Empty title="No ready index to search">Build one in <Link className="text-accent underline" href="/indexes">Indexes &amp; Embeddings</Link> first.</Empty>;
  }
  const res = run.data;
  const cmp = compare.data;
  const ans = ask.data;
  const canRun = query.trim().length > 0 && !!indexId;

  return (
    <div className="mx-auto max-w-[1500px]">
      <PageHeader title="Retrieval Lab" subtitle="Ask a question and see which chunks come back, what each retriever contributed, and where the text lives in the source document." />
      <div className="grid gap-5 xl:grid-cols-[22rem_minmax(0,1fr)]">
        <div className="space-y-4">
          <Card title="Query">
            <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); if (canRun) { setPicked(null); compare.reset(); ask.reset(); run.mutate(); } }}>
              <Field label="Index">
                <Select value={indexId} onChange={(e) => setIndexId(e.target.value)} aria-label="Index">
                  {ready.map((i) => <option key={i.id} value={i.id}>{i.name} ({i.dimension}d)</option>)}
                </Select>
              </Field>
              <Field label="Question">
                <textarea value={query} onChange={(e) => setQuery(e.target.value)} rows={3} aria-label="Question"
                  placeholder="e.g. What are the entrance criteria for the Preliminary Design Review?"
                  className="w-full rounded-md border border-border bg-surface p-2 text-sm placeholder:text-dim"
                  onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canRun) run.mutate(); }} />
              </Field>
              <fieldset>
                <legend className="mb-1 text-xs font-medium text-dim">Retrieval method</legend>
                <div className="space-y-1">
                  {MODES.map((m) => (
                    <label key={m.id} className={clsx("flex cursor-pointer gap-2 rounded-md border px-2.5 py-1.5 text-sm", mode === m.id ? "border-accent bg-accent-soft" : "border-border")}>
                      <input type="radio" name="mode" className="mt-1" checked={mode === m.id} onChange={() => setMode(m.id)} />
                      <span><span className="font-medium">{m.label}</span><span className="block text-xs text-dim">{m.blurb}</span></span>
                    </label>
                  ))}
                </div>
              </fieldset>
              <div className="grid grid-cols-2 gap-2">
                <Field label="Results (top-k)"><Input type="number" min={1} max={50} value={topK} onChange={(e) => setTopK(Number(e.target.value))} /></Field>
                <Field label="Min score" hint="Optional"><Input type="number" step="0.05" value={threshold} onChange={(e) => setThreshold(e.target.value)} placeholder="none" /></Field>
                {mode === "hybrid" && <>
                  <Field label="Candidates each"><Input type="number" min={1} max={200} value={candidates} onChange={(e) => setCandidates(Number(e.target.value))} /></Field>
                  <Field label="RRF k"><Input type="number" min={1} max={1000} value={rrfK} onChange={(e) => setRrfK(Number(e.target.value))} /></Field>
                </>}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Field label="Only document">
                  <Select value={docFilter} onChange={(e) => setDocFilter(e.target.value)} aria-label="Only document">
                    <option value="">All</option>
                    {docs.data?.items.map((d) => <option key={d.id} value={d.id}>{d.filename}</option>)}
                  </Select>
                </Field>
                <Field label="Chunk type">
                  <Select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)} aria-label="Chunk type">
                    <option value="">Text and tables</option><option value="text">Text only</option><option value="table">Tables only</option>
                  </Select>
                </Field>
              </div>
              <div className="flex gap-2">
                <Button type="submit" variant="primary" className="flex-1" loading={run.isPending} disabled={!canRun}><Search className="size-4" aria-hidden /> Search</Button>
                <Button type="button" loading={compare.isPending} disabled={!canRun} onClick={() => { run.reset(); ask.reset(); setPicked(null); compare.mutate(); }}>
                  <GitCompare className="size-4" aria-hidden /> Compare modes
                </Button>
              </div>
              <ErrorBox error={run.error ?? compare.error ?? ask.error} />
            </form>
          </Card>
          <Card title="Advanced retrieval (optional)">
            <div className="space-y-3">
              <Field label="Query transformation" hint="Rewrites or expands the question before searching. Uses the chat model (costs tokens).">
                <Select value={transform} onChange={(e) => setTransform(e.target.value)} aria-label="Query transformation">
                  <option value="none">None</option><option value="rewrite">Rewrite (replaces the question)</option>
                  <option value="multi_query">Multi-query (3 variants, fused)</option><option value="hyde">HyDE (embed a hypothetical answer)</option>
                  <option value="decompose">Decompose (sub-questions)</option>
                </Select>
              </Field>
              <Field label="Reranking" hint="The chat model scores each candidate 0-10 and reorders. Original scores are kept.">
                <Select value={rerank} onChange={(e) => setRerank(e.target.value)} aria-label="Reranking">
                  <option value="none">None</option><option value="llm">LLM reranker</option>
                </Select>
              </Field>
              <label className="flex items-start gap-2 text-sm">
                <input type="checkbox" className="mt-1" checked={mmrOn} onChange={(e) => setMmrOn(e.target.checked)} />
                <span>Diversify results (MMR)<span className="block text-xs text-dim">Skip near-duplicate chunks. No LLM needed.</span></span>
              </label>
              {mmrOn && <Field label={`Relevance vs diversity: ${mmrLambda}`} hint="1 = only relevance, 0 = only diversity.">
                <input type="range" min={0} max={1} step={0.05} value={mmrLambda} onChange={(e) => setMmrLambda(Number(e.target.value))} className="w-full" aria-label="MMR lambda" />
              </Field>}
              {(transform !== "none" || rerank !== "none") && !gen?.configured && <p className="text-xs text-warn">These options need the chat model: {gen?.detail}</p>}
            </div>
          </Card>
          <Card title="Answer with citations">
            {!gen?.configured ? <p className="text-sm text-dim">Not available: {gen?.detail ?? "checking…"}</p> : (
              <div className="space-y-3">
                <Field label="Chat model (deployment)">
                  <Select value={model} onChange={(e) => setModel(e.target.value)} aria-label="Chat model">
                    {gen.models.map((m) => <option key={m}>{m}</option>)}
                  </Select>
                </Field>
                <label className="flex items-start gap-2 text-sm">
                  <input type="checkbox" className="mt-1" checked={strict} onChange={(e) => setStrict(e.target.checked)} />
                  <span>Grounded mode<span className="block text-xs text-dim">Answer only from the retrieved passages, otherwise refuse.</span></span>
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <Field label="Max answer tokens" hint="Reasoning models (gpt-5) need more."><Input type="number" min={16} value={maxOut} onChange={(e) => setMaxOut(Number(e.target.value))} /></Field>
                  <Field label="Context budget" hint="Tokens of passages sent."><Input type="number" min={20} value={budget} onChange={(e) => setBudget(Number(e.target.value))} /></Field>
                </div>
                <Field label="Refuse without asking the model below score" hint="Optional, dense cosine. Saves tokens on off-topic questions.">
                  <Input type="number" step="0.05" value={abstainBelow} onChange={(e) => setAbstainBelow(e.target.value)} placeholder="none" />
                </Field>
                <Button type="button" variant="primary" className="w-full" loading={ask.isPending} disabled={!canRun}
                  onClick={() => { run.reset(); compare.reset(); setPicked(null); ask.mutate(); }}>
                  <MessageSquareText className="size-4" aria-hidden /> Ask with citations
                </Button>
              </div>
            )}
          </Card>
          {res?.info.query_embedding && (
            <Card title="Query embedding">
              <dl className="space-y-1 font-mono text-xs">
                <div className="flex justify-between"><dt className="text-dim">model</dt><dd>{res.info.query_embedding.provider} / {res.info.query_embedding.model}</dd></div>
                <div className="flex justify-between"><dt className="text-dim">dimension</dt><dd>{res.info.query_embedding.dimension}</dd></div>
                <div className="flex justify-between"><dt className="text-dim">norm</dt><dd>{res.info.query_embedding.norm}</dd></div>
                <div className="flex justify-between"><dt className="text-dim">tokens billed</dt><dd>{res.info.query_embedding.tokens_billed}{res.info.query_embedding.cached ? " (cached)" : ""}</dd></div>
              </dl>
              <p className="mt-2 break-words font-mono text-[11px] text-dim">[{res.info.query_embedding.first_values.join(", ")}, …]</p>
              <p className="mt-2 text-[11px] text-dim">The question is embedded with the index&apos;s own model. A different model or size would be refused, not silently compared.</p>
            </Card>
          )}
          {res && <Card title="Timings"><ul className="font-mono text-xs">{Object.entries(res.timings_ms).map(([k, v]) => <li key={k} className="flex justify-between"><span className="text-dim">{k}</span><span>{v} ms</span></li>)}</ul></Card>}
        </div>

        <div className="min-w-0 space-y-4">
          {!res && !cmp && !ans && !run.isPending && !compare.isPending && !ask.isPending && <Empty title="Nothing searched yet">Type a question and press <strong>Search</strong>, <strong>Compare modes</strong> to see all three side by side, or <strong>Ask with citations</strong> for a grounded answer.</Empty>}
          {(run.isPending || compare.isPending) && <Spinner label="Searching" />}
          {ask.isPending && <Spinner label="Retrieving passages and asking the model" />}
          {ans && <AnswerPanel r={ans} onOpen={setPicked} />}
          {ans && ans.context_passages.length > 0 && (
            <Card title={`Passages sent to the model (${ans.context_passages.length})`} pad={false}>
              <ol className="divide-y divide-border">
                {ans.context_passages.map((p) => (
                  <li key={p.chunk_id}>
                    <button onClick={() => setPicked(p)} className={clsx("flex w-full gap-3 px-4 py-2 text-left text-sm hover:bg-surface-2", picked?.text === p.text && "bg-accent-soft")}>
                      <span className="font-mono text-xs font-semibold text-accent">[{p.n}]</span>
                      <span className="min-w-0"><span className="block font-medium">{p.citation}</span><span className="line-clamp-2 text-xs text-dim">{p.text}</span></span>
                    </button>
                  </li>
                ))}
              </ol>
            </Card>
          )}

          {res?.stages && <StageTrace res={res} />}
          {res && (
            <Card title={`${res.results.length} result${res.results.length === 1 ? "" : "s"} · ${MODES.find((m) => m.id === res.mode)?.label}`}>
              {res.info.query_terms && <p className="mb-2 text-xs text-dim">Keyword terms used: {res.info.query_terms.join(", ")}</p>}
              {res.results.length === 0 ? <p className="text-sm text-dim">No chunk matched{threshold ? " above the minimum score" : ""}.</p> : (
                <ol className="space-y-2">
                  {res.results.map((h) => (
                    <li key={h.chunk_id}>
                      <button onClick={() => setPicked(h)} aria-label={`Open result ${h.rank} in its source`}
                        className={clsx("w-full rounded-lg border p-3 text-left hover:bg-surface-2", picked?.text === h.text ? "border-accent bg-accent-soft" : "border-border")}>
                        <div className="mb-1 flex flex-wrap items-center gap-2">
                          <span className="grid size-6 place-items-center rounded bg-text font-mono text-xs text-bg">{h.rank}</span>
                          <span className="text-sm font-medium">{h.citation}</span>
                          {h.type === "table" && <Badge tone="accent">table</Badge>}
                          <span className="ml-auto font-mono text-xs text-dim">score {h.score}</span>
                        </div>
                        {h.section && <p className="mb-1 truncate text-xs text-dim">{h.section}</p>}
                        <p className="line-clamp-3 whitespace-pre-wrap text-sm">{h.text}</p>
                        <div className="mt-2"><Contributions h={h} mode={res.mode} /></div>
                      </button>
                    </li>
                  ))}
                </ol>
              )}
            </Card>
          )}

          {cmp && <CompareModes cmp={cmp} onPick={setPicked} />}
          {picked && <SourcePanel hit={picked} onClose={() => setPicked(null)} />}
        </div>
      </div>
    </div>
  );
}

function CompareModes({ cmp, onPick }: { cmp: Record<Mode, SearchResponse>; onPick: (h: SearchHit) => void }) {
  const seen: Record<string, Mode[]> = {};
  (Object.keys(cmp) as Mode[]).forEach((m) => cmp[m].results.forEach((h) => { (seen[h.chunk_id] ??= []).push(m); }));
  return (
    <Card title="Same question, three retrievers">
      <div className="grid gap-3 lg:grid-cols-3">
        {MODES.map((m) => (
          <div key={m.id}>
            <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">{m.label}</h3>
            <ol className="space-y-1.5">
              {cmp[m.id].results.map((h) => {
                const also = seen[h.chunk_id].filter((x) => x !== m.id);
                return (
                  <li key={h.chunk_id}>
                    <button onClick={() => onPick(h)} className="w-full rounded-md border border-border p-2 text-left text-xs hover:bg-surface-2">
                      <div className="flex items-center gap-2"><span className="font-mono">#{h.rank}</span><span className="truncate font-medium">{h.citation}</span></div>
                      <p className="mt-0.5 line-clamp-2 text-dim">{h.text}</p>
                      <p className="mt-1 text-[11px]">{also.length ? <span className="text-good">also in {also.join(", ")}</span> : <span className="text-warn">only found by {m.id}</span>}</p>
                    </button>
                  </li>
                );
              })}
              {cmp[m.id].results.length === 0 && <li className="text-xs text-dim">No results.</li>}
            </ol>
          </div>
        ))}
      </div>
      <p className="mt-3 text-xs text-dim">Chunks found by only one retriever show where the methods differ. Which is better depends on your questions; measure it before choosing.</p>
    </Card>
  );
}

function StageTrace({ res }: { res: SearchResponse }) {
  return (
    <Card title="Pipeline stages" actions={<span className="text-xs text-dim">{res.llm_tokens ? `${res.llm_tokens} LLM tokens` : "no LLM tokens"}</span>}>
      <ol className="space-y-2 text-sm">
        {res.stages!.map((s) => (
          <li key={s.stage} className="rounded-md border border-border p-2">
            <div className="flex items-center justify-between"><span className="font-medium">{s.stage.replace("_", " ")}{s.method && s.method !== "none" ? ` · ${s.method}` : ""}</span>
              <span className="font-mono text-xs text-dim">{s.ms} ms{s.llm_tokens ? ` · ${s.llm_tokens} tok` : ""}</span></div>
            {s.stage === "query_transform" && s.queries && s.queries.length > 0 && (
              <ul className="mt-1 space-y-0.5 text-xs">{s.queries.map((q, i) => (
                <li key={i} className={q.search ? "" : "text-dim line-through"}><span className="font-mono text-dim">{q.role}{q.mode ? `/${q.mode}` : ""}:</span> {q.text.slice(0, 160)}</li>))}</ul>
            )}
            {s.stage === "retrieve" && s.per_query && <p className="mt-1 text-xs text-dim">{s.per_query.map((q) => `${q.role} (${q.mode}): ${q.n}`).join(" · ")}</p>}
            {s.stage === "fuse" && <p className="mt-1 text-xs text-dim">{s.n_queries} search(es) → {s.unique_chunks} unique chunks</p>}
            {s.stage === "mmr" && <p className="mt-1 text-xs text-dim">λ {s.lambda} · {s.changed_top_k ? "changed the top results" : "top results unchanged"}</p>}
            {s.stage === "rerank" && <p className="mt-1 text-xs text-dim">{s.pool} candidates scored{s.warning ? ` · ${s.warning}` : ""}</p>}
          </li>
        ))}
      </ol>
      {res.warnings && res.warnings.length > 0 && <p className="mt-2 text-xs text-warn">{res.warnings.join(" ")}</p>}
    </Card>
  );
}
