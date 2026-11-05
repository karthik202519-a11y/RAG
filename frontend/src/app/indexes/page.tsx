"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Database, Search, Trash2 } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import { ParamControl, defaults, validate } from "@/components/param-form";
import { Badge, Button, Card, Empty, ErrorBox, Field, Input, PageHeader, Select, Spinner, fmt, statusTone } from "@/components/ui";
import { api, type Catalog, type Doc, type IndexInfo, type Providers } from "@/lib/api";
import { useProject } from "@/lib/project";

const STORE_HINT: Record<string, string> = {
  qdrant: "Approximate (HNSW) search with payload filters.", faiss: "Exact brute-force search; no server, saved to disk.",
  chroma: "Embedded HNSW database on disk.", lancedb: "Embedded columnar files; exact scan with filters first.",
  pgvector: "HNSW inside PostgreSQL; needs PGVECTOR_URL (pgvector/pgvector Docker image).",
};

export default function IndexesPage() {
  const qc = useQueryClient();
  const { project, loading } = useProject();
  const pid = project?.id;
  const providers = useQuery({ queryKey: ["providers"], staleTime: 60_000, queryFn: () => api<Providers>("/providers") });
  const catalog = useQuery({ queryKey: ["catalog", "chunking"], staleTime: Infinity, queryFn: () => api<Catalog>("/method-catalog?category=chunking") });
  const docs = useQuery({ queryKey: ["docs-ready", pid], enabled: !!pid, queryFn: () => api<{ items: Doc[] }>(`/projects/${pid}/documents?status=ready&limit=500`) });
  const indexes = useQuery({
    queryKey: ["indexes", pid], enabled: !!pid,
    queryFn: () => api<IndexInfo[]>(`/projects/${pid}/indexes`),
    refetchInterval: (q) => (q.state.data?.some((i) => i.status === "building") ? 1500 : 8000),
  });

  const methods = useMemo(() => (catalog.data?.items ?? []).filter((m) => m.status !== "unavailable"), [catalog.data]);
  const [name, setName] = useState("");
  const [providerId, setProviderId] = useState("");
  const [method, setMethod] = useState("structure_aware");
  const [params, setParams] = useState<Record<string, unknown> | null>(null);
  const [excluded, setExcluded] = useState<string[]>([]);
  const [prefix, setPrefix] = useState(true);
  const [store, setStore] = useState("qdrant");
  const [contextual, setContextual] = useState(false);

  const m = methods.find((x) => x.id === method);
  const p = params ?? (m ? defaults(m) : {});
  const available = (providers.data?.embedding ?? []).filter((e) => e.available);
  const provider = providerId || available.find((e) => e.semantic)?.id || available[0]?.id || "";
  const selDocs = (docs.data?.items ?? []).filter((d) => !excluded.includes(d.id));
  const selTokens = selDocs.reduce((a, d) => a + (d.latest_version?.token_count ?? 0), 0);
  const problem = validate(m, p);
  const chosen = providers.data?.embedding.find((e) => e.id === provider);

  const create = useMutation({
    mutationFn: () => api<IndexInfo>(`/projects/${pid}/indexes`, {
      method: "POST",
      json: { name: name || `${m?.name} · ${chosen?.model}`, provider, store, contextual, section_prefix: prefix,
        chunking: { method, params: p }, document_ids: excluded.length ? selDocs.map((d) => d.id) : null },
    }),
    onSuccess: () => { setName(""); void qc.invalidateQueries({ queryKey: ["indexes", pid] }); },
  });
  const del = useMutation({
    mutationFn: (id: string) => api(`/indexes/${id}`, { method: "DELETE" }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["indexes", pid] }),
  });

  if (loading) return <Spinner />;
  if (!project) return <Empty title="No project selected"><Link className="text-accent underline" href="/projects">Create a project first</Link></Empty>;

  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader title="Indexes & Embeddings"
        subtitle="An index is a frozen recipe: which documents, cut by which chunker, embedded by which model, stored in which collection. Change any ingredient and you build a new index. Nothing is overwritten, so you can compare them." />

      <div className="grid gap-5 xl:grid-cols-[26rem_1fr]">
        <div className="space-y-5">
          <Card title="Providers">
            <ul className="space-y-2 text-sm">
              {providers.data?.embedding.map((e) => (
                <li key={e.id} className="flex items-start justify-between gap-2">
                  <span><span className="font-medium">{e.id}</span>{e.model && <span className="text-dim"> · {e.model}</span>}
                    {!e.semantic && <span className="block text-xs text-warn">Offline demo: matches shared words, not meaning.</span>}
                    {!e.available && <span className="block text-xs text-dim">{e.detail}</span>}</span>
                  {!e.available && <Badge tone="warn">not configured</Badge>}
                </li>
              ))}
              {providers.data?.vector_stores.map((v) => (
                <li key={v.id} className="flex items-center justify-between"><span><span className="font-medium">{v.id}</span><span className="text-dim"> · {v.mode}</span></span></li>
              ))}
            </ul>
            <p className="mt-3 text-xs text-dim">Answer generation: {providers.data?.generation.configured ? "configured" : providers.data?.generation.detail}</p>
          </Card>

          <Card title="Build a new index">
            {(docs.data?.items.length ?? 0) === 0 ? (
              <Empty title="No ready documents"><Link className="text-accent underline" href="/ingestion">Upload and parse a document first</Link></Empty>
            ) : (
              <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
                <Field label="Name"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder={`${m?.name ?? "Index"} · ${chosen?.model ?? ""}`} /></Field>
                <Field label="Embedding provider">
                  <Select value={provider} onChange={(e) => setProviderId(e.target.value)} aria-label="Embedding provider">
                    {providers.data?.embedding.map((e) => <option key={e.id} value={e.id} disabled={!e.available}>{e.id}{e.model ? ` (${e.model})` : ""}{e.available ? "" : " - not configured"}</option>)}
                  </Select>
                </Field>
                <Field label="Vector store" hint={STORE_HINT[store] ?? ""}>
                  <Select value={store} onChange={(e) => setStore(e.target.value)} aria-label="Vector store">
                    {(providers.data?.vector_stores ?? [{ id: "qdrant", available: true, mode: "", detail: "" }]).map((v) => <option key={v.id} value={v.id} disabled={!v.available} title={v.detail}>{v.id}{v.mode ? ` (${v.mode})` : ""}{v.available ? "" : " - not configured"}</option>)}
                  </Select>
                </Field>
                <Field label="Chunking method">
                  <Select value={method} onChange={(e) => { setMethod(e.target.value); setParams(null); }} aria-label="Chunking method">
                    {methods.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
                  </Select>
                </Field>
                {m?.params.map((sp) => <ParamControl key={sp.name} p={sp} value={p[sp.name]} onChange={(v) => setParams({ ...p, [sp.name]: v })} />)}
                <label className="flex items-start gap-2 text-sm">
                  <input type="checkbox" className="mt-1" checked={prefix} onChange={(e) => setPrefix(e.target.checked)} />
                  <span>Prefix text chunks with their section title when embedding<span className="block text-xs text-dim">Cheap context: the model also sees which section the text came from.</span></span>
                </label>
                <label className="flex items-start gap-2 text-sm">
                  <input type="checkbox" className="mt-1" checked={contextual} onChange={(e) => setContextual(e.target.checked)} />
                  <span>Contextual embedding<span className="block text-xs text-dim">A chat model writes one sentence placing each text chunk in its document; it is embedded with the chunk and stored so you can read it. Costs one chat call per chunk.</span></span>
                </label>
                <fieldset>
                  <legend className="mb-1 text-xs font-medium text-dim">Documents ({selDocs.length} of {docs.data?.items.length})</legend>
                  <ul className="max-h-36 space-y-1 overflow-auto text-sm">
                    {docs.data?.items.map((d) => (
                      <li key={d.id}><label className="flex items-center gap-2">
                        <input type="checkbox" checked={!excluded.includes(d.id)}
                          onChange={(e) => setExcluded(e.target.checked ? excluded.filter((x) => x !== d.id) : [...excluded, d.id])} />
                        <span className="truncate">{d.filename}</span><span className="ml-auto font-mono text-xs text-dim">{fmt.n(d.latest_version?.token_count)} tok</span>
                      </label></li>
                    ))}
                  </ul>
                  <p className="mt-1 text-xs text-dim">About {fmt.n(selTokens)} tokens of source text. Overlap makes the embedded total slightly larger; the build reports the exact count billed.</p>
                </fieldset>
                {problem && <ErrorBox error={new Error(problem)} />}
                <ErrorBox error={create.error} />
                <Button type="submit" variant="primary" className="w-full" loading={create.isPending} disabled={!!problem || !provider || selDocs.length === 0}>
                  <Database className="size-4" aria-hidden /> Build index
                </Button>
              </form>
            )}
          </Card>
        </div>

        <div className="space-y-4">
          <h2 className="text-sm font-semibold">Indexes ({indexes.data?.length ?? 0})</h2>
          <ErrorBox error={del.error} />
          {indexes.isLoading ? <Spinner /> : indexes.data?.length === 0 ? <Empty title="No indexes yet">Build one on the left, then search it in the Retrieval Lab.</Empty> : (
            indexes.data?.map((i) => <IndexCard key={i.id} i={i} onDelete={() => { if (window.confirm(`Delete index "${i.name}"? Its vectors are removed; your documents are not.`)) del.mutate(i.id); }} />)
          )}
        </div>
      </div>
    </div>
  );
}

function IndexCard({ i, onDelete }: { i: IndexInfo; onDelete: () => void }) {
  const [open, setOpen] = useState(false);
  const rows: [string, string][] = [
    ["Embedding model", `${i.provider} / ${i.model}`],
    ["Vector size", i.dimension ? `${i.dimension} dimensions · ${i.metric} · ${i.store}` : "n/a"],
    ["Chunks / vectors", i.chunk_count != null ? `${fmt.n(i.chunk_count)}` : "n/a"],
    ["Chunker", `${i.chunking.method} ${Object.entries(i.chunking.params).filter(([k]) => ["size", "overlap", "stride"].includes(k)).map(([k, v]) => `${k}=${v}`).join(" ")}`],
    ["Tokens embedded", fmt.n(i.tokens_embedded)],
    ["Tokens billed", i.tokens_billed != null ? `${fmt.n(i.tokens_billed)} (${fmt.n(i.cache_hits)} cache hits)` : "n/a"],
    ["Build time", i.build_seconds != null ? `${i.build_seconds}s` : "n/a"],
    ["Collection", i.collection],
  ];
  return (
    <Card title={<span className="flex items-center gap-2">{i.name} <Badge tone={statusTone(i.status === "building" ? "running" : i.status)}>{i.status}</Badge></span>}
      actions={<>
        {i.status === "ready" && <Link href={`/retrieval?index=${i.id}`}><Button size="sm" variant="primary"><Search className="size-3.5" aria-hidden /> Search</Button></Link>}
        {i.status === "ready" && <Button size="sm" onClick={() => setOpen(!open)}>{open ? "Hide records" : "Inspect"}</Button>}
        <Button size="sm" variant="ghost" onClick={onDelete} aria-label={`Delete ${i.name}`}><Trash2 className="size-3.5" /></Button>
      </>}>
      {i.status === "building" && i.job && (
        <div className="mb-3">
          <div className="h-1.5 overflow-hidden rounded bg-surface-2" role="progressbar" aria-label={`Build progress for ${i.name}`} aria-valuenow={Math.round(i.job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full bg-accent transition-all" style={{ width: `${Math.max(3, i.job.progress * 100)}%` }} />
          </div>
          <p className="mt-1 text-xs text-dim">{i.job.stage ?? "starting"} · attempt {i.job.attempts}</p>
        </div>
      )}
      {i.error && <div className="mb-3"><ErrorBox error={new Error(i.error)} /></div>}
      <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
        {rows.map(([k, v]) => <div key={k} className="flex justify-between gap-3"><dt className="text-dim">{k}</dt><dd className="truncate text-right font-mono text-xs" title={v}>{v}</dd></div>)}
      </dl>
      {open && <Records id={i.id} />}
    </Card>
  );
}

function Records({ id }: { id: string }) {
  const [offset, setOffset] = useState(0);
  const [vec, setVec] = useState<string | null>(null);
  const page = useQuery({ queryKey: ["records", id, offset], queryFn: () => api<{ total: number; items: { id: string; filename: string; chunk_index: number; type: string; tokens: number; pages: number[]; section: string | null; text: string }[] }>(`/indexes/${id}/chunks?limit=8&offset=${offset}`) });
  const v = useQuery({ queryKey: ["vec", id, vec], enabled: !!vec, queryFn: () => api<{ dimension: number; norm: number; min: number; max: number; first_values: number[] }>(`/indexes/${id}/chunks/${vec}/vector`) });
  return (
    <div className="mt-4 border-t border-border pt-3">
      <p className="mb-2 text-xs text-dim">Stored records ({page.data?.total ?? "…"}). Click a row to see its stored vector.</p>
      <ul className="divide-y divide-border text-xs">
        {page.data?.items.map((r) => (
          <li key={r.id}>
            <button className="flex w-full gap-3 py-1.5 text-left hover:bg-surface-2" onClick={() => setVec(vec === r.id ? null : r.id)}>
              <span className="w-28 shrink-0 truncate font-medium">{r.filename}</span>
              <span className="w-8 shrink-0 font-mono">#{r.chunk_index}</span>
              <span className="w-14 shrink-0 font-mono">{r.tokens} tok</span>
              <span className="truncate text-dim">{r.text.replace(/\s+/g, " ").slice(0, 110)}</span>
            </button>
            {vec === r.id && v.data && (
              <div className="mb-2 rounded bg-surface-2 p-2 font-mono">
                <div>dimension {v.data.dimension} · norm {v.data.norm} · min {v.data.min} · max {v.data.max}</div>
                <div className="break-words text-dim">[{v.data.first_values.join(", ")}, …]</div>
              </div>
            )}
          </li>
        ))}
      </ul>
      <div className="mt-2 flex gap-2">
        <Button size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 8))}>Previous</Button>
        <Button size="sm" disabled={offset + 8 >= (page.data?.total ?? 0)} onClick={() => setOffset(offset + 8)}>Next</Button>
      </div>
    </div>
  );
}
