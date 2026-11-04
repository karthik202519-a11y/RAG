"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Play, Plus, Trash2 } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChunkDetail, ChunkedText } from "@/components/chunk-view";
import { MethodDetail } from "@/components/method-detail";
import { Badge, Button, Card, Empty, ErrorBox, Field, Input, PageHeader, Select, Spinner } from "@/components/ui";
import { ParamControl, defaults, validate } from "@/components/param-form";
import { api, type Catalog, type ChunkPreview, type Doc } from "@/lib/api";
import { useProject } from "@/lib/project";

type Cfg = { id: number; method: string; params: Record<string, unknown>; label: string };

const PRESETS: { label: string; method: string; params: Record<string, unknown> }[] = [
  { label: "200 tok · 0 overlap", method: "fixed_token", params: { size: 200, overlap: 0 } },
  { label: "200 tok · 40 overlap", method: "fixed_token", params: { size: 200, overlap: 40 } },
  { label: "400 tok · 40 overlap", method: "recursive", params: { size: 400, overlap: 40 } },
  { label: "400 tok · 80 overlap", method: "recursive", params: { size: 400, overlap: 80 } },
  { label: "800 tok · 80 overlap", method: "recursive", params: { size: 800, overlap: 80 } },
  { label: "800 tok · 160 overlap", method: "recursive", params: { size: 800, overlap: 160 } },
];
const NOT_YET = ["Semantic", "Parent-child", "Contextual"];

export default function ChunkingPlayground() {
  const { project, loading } = useProject();
  const pid = project?.id;
  // User choices are stored only when they differ from a derived default (first ready document,
  // recursive chunking, a small page range for big documents), so no effect has to sync state.
  const [docIdState, setDocId] = useState<string | null>(null);
  const [rangeState, setRangeState] = useState<{ docId: string; from: string; to: string } | null>(null);
  const [cfgsState, setCfgs] = useState<Cfg[] | null>(null);
  const [active, setActive] = useState(0);
  const [selected, setSelected] = useState<{ cfg: number; chunk: number } | null>(null);
  const [view, setView] = useState<"text" | "list" | "compare">("text");
  const [price, setPrice] = useState("");
  const [nextId, setNextId] = useState(2);

  const catalog = useQuery({ queryKey: ["catalog", "chunking"], queryFn: () => api<Catalog>("/method-catalog?category=chunking") });
  const docs = useQuery({
    queryKey: ["docs-ready", pid], enabled: !!pid,
    queryFn: () => api<{ items: Doc[] }>(`/projects/${pid}/documents?status=ready&limit=500`),
  });
  const methods = useMemo(() => catalog.data?.items ?? [], [catalog.data]);
  const byId = useMemo(() => Object.fromEntries(methods.map((m) => [m.id, m])), [methods]);

  const initialCfgs = useMemo<Cfg[]>(() => {
    const m = byId["recursive"] ?? methods.find((x) => x.status !== "unavailable");
    return m ? [{ id: 1, method: m.id, params: defaults(m), label: "A" }] : [];
  }, [methods, byId]);
  const cfgs = cfgsState ?? initialCfgs;

  // document from ?doc= (the Chunk link in Ingestion) or the first ready one
  const preferredDoc = useMemo(() => {
    if (!docs.data) return "";
    const want = typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("doc");
    return (docs.data.items.find((d) => d.id === want) ?? docs.data.items[0])?.id ?? "";
  }, [docs.data]);
  const docId = docIdState ?? preferredDoc;

  const doc = docs.data?.items.find((d) => d.id === docId);
  const pages = doc?.latest_version?.pages ?? 0;
  const BIG = 15;
  // large documents start on a small page range so the first run is fast and readable
  const range = rangeState?.docId === docId ? rangeState : { docId, from: pages > BIG ? "1" : "", to: pages > BIG ? "8" : "" };
  const setRange = (r: { from: string; to: string }) => setRangeState({ docId, ...r });

  const [previewState, setPreview] = useState<ChunkPreview | null>(null);
  const preview = previewState?.document_id === docId ? previewState : null;
  const run = useMutation({
    mutationFn: () => api<ChunkPreview>("/chunking-preview", {
      method: "POST",
      json: {
        document_id: docId, page_from: range.from ? Number(range.from) : null, page_to: range.to ? Number(range.to) : null,
        price_per_million_tokens: price ? Number(price) : null,
        configs: cfgs.map((c) => ({ method: c.method, params: c.params, label: `${c.label}: ${byId[c.method]?.name ?? c.method}` })),
      },
    }),
    onSuccess: (r) => { setPreview(r); setSelected(null); },
  });

  const cur = cfgs[active];
  const curMethod = cur ? byId[cur.method] : undefined;
  const problem = cur ? validate(curMethod, cur.params) : null;
  const allProblems = cfgs.map((c) => validate(byId[c.method], c.params)).find(Boolean) ?? null;
  const update = (patch: Partial<Cfg>) => setCfgs(cfgs.map((c, i) => (i === active ? { ...c, ...patch } : c)));
  const setMethod = (id: string) => { const m = byId[id]; if (m) update({ method: id, params: defaults(m) }); };
  const addConfig = (method?: string, params?: Record<string, unknown>) => {
    if (cfgs.length >= 4) return;
    const m = byId[method ?? cur?.method ?? "recursive"];
    if (!m) return;
    setCfgs([...cfgs, { id: nextId, method: m.id, params: params ?? defaults(m), label: String.fromCharCode(65 + cfgs.length) }]);
    setNextId(nextId + 1); setActive(cfgs.length);
  };
  const applyPreset = (p: (typeof PRESETS)[number]) => {
    if (cur) update({ method: p.method, params: { ...defaults(byId[p.method]), ...p.params } });
  };

  if (loading || catalog.isLoading) return <Spinner />;
  if (!project) return <Empty title="No project selected"><Link className="text-accent underline" href="/projects">Create a project first</Link></Empty>;

  const results = preview?.results ?? [];
  const selRes = selected ? results[selected.cfg] : null;
  const selChunk = selRes && selected ? selRes.chunks[selected.chunk] : null;

  return (
    <div className="mx-auto max-w-[1600px]">
      <PageHeader title="Chunking Playground"
        subtitle="Cut a real document with different methods and see exactly where each chunk starts and ends. Every number comes from running the method on your document."
        actions={
          <Button variant="primary" loading={run.isPending} disabled={!docId || !!allProblems || cfgs.length === 0} onClick={() => run.mutate()}>
            <Play className="size-4" aria-hidden /> Run {cfgs.length > 1 ? `${cfgs.length} configurations` : "chunking"}
          </Button>
        } />

      <div className="mb-4 grid gap-3 rounded-lg border border-border bg-surface p-3 md:grid-cols-[1fr_auto_auto_auto]">
        <Field label="Document">
          <Select value={docId} onChange={(e) => setDocId(e.target.value)} aria-label="Document">
            {docs.data?.items.length === 0 && <option value="">No ready documents</option>}
            {docs.data?.items.map((d) => <option key={d.id} value={d.id}>{d.filename} ({d.latest_version?.pages} p, {d.latest_version?.token_count?.toLocaleString()} tok)</option>)}
          </Select>
        </Field>
        <Field label="From page"><Input type="number" min={1} max={pages || undefined} className="w-24" value={range.from} placeholder="1" onChange={(e) => setRange({ ...range, from: e.target.value })} /></Field>
        <Field label="To page"><Input type="number" min={1} max={pages || undefined} className="w-24" value={range.to} placeholder={String(pages || "")} onChange={(e) => setRange({ ...range, to: e.target.value })} /></Field>
        <Field label="Embedding price ($ / 1M tokens)" hint="Optional. Needed only for the cost estimate."><Input type="number" min={0} step="0.001" className="w-40" value={price} placeholder="e.g. 0.02" onChange={(e) => setPrice(e.target.value)} /></Field>
      </div>
      {pages > BIG && <p className="mb-3 text-xs text-dim">This document has {pages} pages, so the range starts at 1–8. Clear the fields to chunk everything (slower, and the text view shows at most 120,000 characters).</p>}
      {docs.data?.items.length === 0 && <Empty title="No ready documents in this project"><Link className="text-accent underline" href="/ingestion">Upload and parse a document first</Link></Empty>}

      <div className="grid gap-4 xl:grid-cols-[17rem_minmax(0,1fr)_24rem]">
        {/* ---- left: method catalog ---- */}
        <Card title="Methods" pad={false} className="h-fit">
          <ul className="max-h-[75vh] divide-y divide-border overflow-y-auto" aria-label="Chunking methods">
            {[...methods].sort((a, b) => Number(a.status === "unavailable") - Number(b.status === "unavailable")).map((m) => (
              <li key={m.id}>
                <button onClick={() => (m.status === "unavailable" ? undefined : setMethod(m.id))} disabled={m.status === "unavailable"}
                  aria-label={`${m.name}${m.status === "unavailable" ? " (not built yet)" : ""}`}
                  title={m.unavailable_reason ?? m.summary}
                  className={clsx("flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm",
                    m.status === "unavailable" ? "cursor-not-allowed text-dim opacity-60" : "hover:bg-surface-2",
                    cur?.method === m.id && "bg-accent-soft font-medium")}>
                  <span>{m.name}</span>
                  {m.status === "unavailable" && <Badge tone="warn">soon</Badge>}
                </button>
              </li>
            ))}
          </ul>
        </Card>

        {/* ---- centre: visualisation ---- */}
        <div className="min-w-0 space-y-4">
          <div className="flex flex-wrap items-center gap-2" role="tablist" aria-label="Configurations">
            {cfgs.map((c, i) => (
              <button key={c.id} role="tab" aria-selected={i === active} onClick={() => setActive(i)}
                className={clsx("flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm", i === active ? "border-accent bg-accent-soft font-medium" : "border-border bg-surface hover:bg-surface-2")}>
                <span className="font-mono text-xs">{c.label}</span> {byId[c.method]?.name}
                <span className="font-mono text-[11px] text-dim">{String(c.params.size ?? "")}{c.params.overlap != null ? `/${c.params.overlap}` : ""}</span>
              </button>
            ))}
            <Button size="sm" onClick={() => addConfig()} disabled={cfgs.length >= 4} aria-label="Add a configuration to compare"><Plus className="size-3.5" /> Compare</Button>
            {cfgs.length > 1 && (
              <Button size="sm" variant="ghost" aria-label={`Remove configuration ${cur?.label}`}
                onClick={() => { setCfgs(cfgs.filter((_, i) => i !== active).map((c, i) => ({ ...c, label: String.fromCharCode(65 + i) }))); setActive(0); setPreview(null); }}>
                <Trash2 className="size-3.5" /> Remove {cur?.label}
              </Button>
            )}
          </div>

          <ErrorBox error={run.error} />
          {run.isPending && <Spinner label="Chunking the document" />}

          {!preview && !run.isPending && (
            <Empty title="Nothing run yet">Choose a document and a method, then press <strong>Run</strong>. Results appear here with exact chunk boundaries.</Empty>
          )}

          {preview && (
            <>
              <div className="flex gap-1 rounded-md bg-surface-2 p-1 text-sm" role="tablist" aria-label="View">
                {([["text", "Document view"], ["list", "Chunk list"], ["compare", "Statistics & comparison"]] as const).map(([k, l]) => (
                  <button key={k} role="tab" aria-selected={view === k} onClick={() => setView(k)}
                    className={clsx("flex-1 rounded px-3 py-1", view === k ? "bg-surface font-medium shadow-sm" : "text-dim")}>{l}</button>
                ))}
              </div>
              <p className="text-xs text-dim">{preview.filename} · pages {preview.page_from}–{preview.page_to} · {preview.doc_text.length.toLocaleString()} characters</p>

              {view === "text" && (
                <div className={clsx("grid gap-4", results.length > 1 && "lg:grid-cols-2")}>
                  {results.map((r, ci) => (
                    <Card key={ci} title={<span>{r.label} <span className="font-normal text-dim">· {r.stats.chunk_count} chunks</span></span>} pad>
                      {r.warnings.map((w) => <p key={w} className="mb-2 rounded bg-warn-soft px-2 py-1 text-xs text-warn">{w}</p>)}
                      <div className="max-h-[60vh] overflow-auto">
                        <ChunkedText text={preview.doc_text} chunks={r.chunks} selected={selected?.cfg === ci ? selected.chunk : null}
                          onSelect={(i) => setSelected({ cfg: ci, chunk: i })} />
                      </div>
                      <p className="mt-2 text-[11px] text-dim">Colors alternate between neighbouring chunks. Hatched text is repeated in two chunks (overlap). Struck-through text is in no chunk. Click text to inspect its chunk.</p>
                    </Card>
                  ))}
                </div>
              )}

              {view === "list" && (
                <div className={clsx("grid gap-4", results.length > 1 && "lg:grid-cols-2")}>
                  {results.map((r, ci) => (
                    <Card key={ci} title={r.label} pad={false}>
                      <ul className="max-h-[60vh] divide-y divide-border overflow-auto">
                        {r.chunks.map((c) => (
                          <li key={c.index}>
                            <button onClick={() => setSelected({ cfg: ci, chunk: c.index })}
                              className={clsx("flex w-full items-center gap-3 px-3 py-1.5 text-left text-xs hover:bg-surface-2", selected?.cfg === ci && selected.chunk === c.index && "bg-accent-soft")}>
                              <span className="size-3 shrink-0 rounded-sm" style={{ background: `var(--c${c.index % 8})` }} />
                              <span className="w-10 font-mono">#{c.index}</span>
                              <span className="w-16 font-mono tabular-nums">{c.tokens} tok</span>
                              <span className="w-16 text-dim">p. {c.pages.join(",") || "-"}</span>
                              {c.type === "table" && <Badge tone="accent">table</Badge>}
                              <span className="truncate text-dim">{(c.text ?? preview.doc_text.slice(c.start ?? 0, c.end ?? 0)).slice(0, 80).replace(/\s+/g, " ")}</span>
                            </button>
                          </li>
                        ))}
                      </ul>
                    </Card>
                  ))}
                </div>
              )}

              {view === "compare" && <Compare results={results} />}

              {selChunk && selRes && (
                <Card title={`Selected chunk · ${selRes.label}`} actions={<Button size="sm" variant="ghost" onClick={() => setSelected(null)}>Close</Button>}>
                  <ChunkDetail chunk={selChunk} docText={preview.doc_text} />
                </Card>
              )}
            </>
          )}
        </div>

        {/* ---- right: configuration and explanation ---- */}
        <div className="space-y-4">
          <Card title={`Configure ${cur?.label ?? ""}`}>
            <div className="mb-3">
              <p className="mb-1 text-xs font-medium text-dim">Experiment presets</p>
              <div className="flex flex-wrap gap-1.5">
                {PRESETS.map((p) => <Button key={p.label} size="sm" onClick={() => applyPreset(p)}>{p.label}</Button>)}
                {NOT_YET.map((n) => <Button key={n} size="sm" disabled title="Not built yet; see the Method Catalog">{n}</Button>)}
              </div>
              <p className="mt-1 text-[11px] text-dim">Presets are experiments to try on your data, not recommendations.</p>
            </div>
            {curMethod && (
              <div className="space-y-3">
                <Field label="Method">
                  <Select value={cur.method} onChange={(e) => setMethod(e.target.value)} aria-label="Method">
                    {methods.filter((m) => m.status !== "unavailable").map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
                  </Select>
                </Field>
                {curMethod.params.map((p) => (
                  <ParamControl key={p.name} p={p} value={cur.params[p.name]} onChange={(v) => update({ params: { ...cur.params, [p.name]: v } })} />
                ))}
                {problem && <ErrorBox error={new Error(problem)} />}
              </div>
            )}
          </Card>
          <Card>{curMethod ? <MethodDetail m={curMethod} names={Object.fromEntries(methods.map((m) => [m.id, m.name]))} /> : <Spinner />}</Card>
        </div>
      </div>
    </div>
  );
}

function Compare({ results }: { results: ChunkPreview["results"] }) {
  const rows: [string, (r: ChunkPreview["results"][number]) => string, string?][] = [
    ["Chunks", (r) => r.stats.chunk_count.toLocaleString()],
    ["Text / table chunks", (r) => `${r.stats.text_chunks} / ${r.stats.table_chunks}`],
    ["Tokens per chunk (median)", (r) => String(r.stats.tokens_median)],
    ["Tokens per chunk (p10 – p90)", (r) => `${r.stats.tokens_p10} – ${r.stats.tokens_p90}`],
    ["Smallest / largest chunk", (r) => `${r.stats.tokens_min} / ${r.stats.tokens_max}`],
    ["Duplication from overlap", (r) => `${r.stats.duplication_ratio.toFixed(3)}×`, "1.000× means no text is repeated; 1.200× means 20% more text is stored."],
    ["Text covered by chunks", (r) => `${(r.stats.coverage_ratio * 100).toFixed(1)}%`, "Below 100% means some text is in no chunk."],
    ["Text chunks ending mid-sentence", (r) => `${r.stats.mid_sentence_pct}%`],
    ["Tokens to embed", (r) => r.stats.estimated_embedding_tokens.toLocaleString()],
    ["Estimated embedding cost", (r) => (r.stats.estimated_embedding_cost_usd == null ? "enter a price above" : `$${r.stats.estimated_embedding_cost_usd.toFixed(5)}`)],
    ["Vectors to store", (r) => r.stats.vectors_to_store.toLocaleString()],
    ["Chunking time", (r) => `${(r.stats.processing_seconds * 1000).toFixed(0)} ms`],
  ];
  return (
    <div className="space-y-4">
      <Card title="Comparison" pad={false}>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border text-xs text-dim">
              <tr><th className="px-4 py-2 font-medium">Metric</th>{results.map((r, i) => <th key={i} className="px-3 font-medium">{r.label}</th>)}</tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map(([label, f, note]) => (
                <tr key={label}>
                  <td className="px-4 py-1.5" title={note}>{label}</td>
                  {results.map((r, i) => <td key={i} className="px-3 font-mono tabular-nums">{f(r)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className={clsx("grid gap-4", results.length > 1 && "lg:grid-cols-2")}>
        {results.map((r, i) => (
          <Card key={i} title={`Chunk size distribution · ${r.label}`}>
            <div className="h-48" aria-label={`Histogram of chunk token counts for ${r.label}`}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={r.stats.histogram.map((h) => ({ range: `${h.from}–${h.to}`, count: h.count }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="range" tick={{ fontSize: 10, fill: "var(--text-dim)" }} interval="preserveStartEnd" minTickGap={28}
                    tickFormatter={(v: string) => v.split("–")[0]} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: "var(--text-dim)" }} />
                  <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12 }} />
                  <Bar dataKey="count" fill="var(--accent)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-1 text-[11px] text-dim">Tokens per chunk (x) against how many chunks fall in that range (y).</p>
          </Card>
        ))}
      </div>
      <Card title="How to read this">
        <ul className="list-disc space-y-1 pl-5 text-sm text-dim">
          <li><strong className="text-text">Smaller chunks</strong> give more precise matches but each carries less context, and you store more vectors.</li>
          <li><strong className="text-text">Overlap</strong> protects facts that straddle a boundary, at the price of storing repeated text (see duplication).</li>
          <li><strong className="text-text">Mid-sentence endings</strong> are a symptom of structure-blind cutting, not proof of worse retrieval. Retrieval quality is measured in the Evaluation Center, which is not built yet.</li>
        </ul>
      </Card>
    </div>
  );
}
