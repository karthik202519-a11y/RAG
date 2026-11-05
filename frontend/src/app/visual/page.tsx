"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { useState } from "react";
import { Badge, Button, Empty, ErrorBox, Field, Input, PageHeader, Select, Spinner } from "@/components/ui";
import { ChunkText } from "@/components/rich-text";
import { api, apiBlobUrl, type Doc } from "@/lib/api";
import { useProject } from "@/lib/project";

type Fig = { n: number; kind: string; bbox: number[]; width_pt: number; height_pt: number; share_of_page: number; image_url: string };
type Chart = { is_chart: boolean; markdown: string; model: string; tokens: number; warning: string };
type PageHit = { page: number; score: number; image_url: string };
type PageIdx = { available: boolean; detail: string; indexed_pages: number; pages: number | null };

function AuthImg({ path, alt, className }: { path: string; alt: string; className?: string }) {
  const q = useQuery({ queryKey: ["img", path], queryFn: () => apiBlobUrl(path), staleTime: Infinity, gcTime: 5 * 60_000 });
  if (!q.data) return <div className={clsx("grid place-items-center bg-surface-2 text-xs text-dim", className)} style={{ minHeight: 96 }}>{q.isError ? "Image failed to load" : "Loading image…"}</div>;
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={q.data} alt={alt} className={className} />;
}

function Crops({ docId, pages }: { docId: string; pages: number }) {
  const [page, setPage] = useState(1);
  const [asked, setAsked] = useState<number | null>(null);
  const figs = useQuery({ queryKey: ["crops", docId, asked], enabled: asked != null, queryFn: () => api<{ figures: Fig[] }>(`/documents/${docId}/figure-crops?page=${asked}`) });
  const chart = useMutation({ mutationFn: (n: number) => api<Chart>(`/documents/${docId}/chart-to-table`, { method: "POST", json: { page: asked, figure: n } }) });
  return (
    <div className="space-y-3">
      <form className="flex items-end gap-3" onSubmit={(e) => { e.preventDefault(); setAsked(page); chart.reset(); }}>
        <Field label="Page" hint={`1 to ${pages}`}><Input type="number" min={1} max={pages} className="w-28" value={page} onChange={(e) => setPage(Number(e.target.value))} /></Field>
        <Button type="submit" disabled={page < 1 || page > pages}>Find figures</Button>
      </form>
      <ErrorBox error={figs.error ?? chart.error} />
      {figs.isLoading && <Spinner label="Finding figures" />}
      {figs.data && figs.data.figures.length === 0 && <p className="text-sm text-dim">No figure-sized pictures or drawings on page {asked}.</p>}
      <ul className="grid gap-3 md:grid-cols-2">
        {figs.data?.figures.map((f) => (
          <li key={f.n} className="rounded-lg border border-border bg-surface p-3">
            <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-dim"><Badge>{f.kind}</Badge>figure {f.n} · {f.width_pt} x {f.height_pt} pt · {Math.round(f.share_of_page * 100)}% of the page</div>
            <AuthImg path={f.image_url} alt={`Figure ${f.n} on page ${asked}`} className="max-h-72 w-full rounded border border-border object-contain" />
            <div className="mt-2"><Button size="sm" onClick={() => chart.mutate(f.n)} disabled={chart.isPending}>{chart.isPending && chart.variables === f.n ? "Reading…" : "Read as a table"}</Button></div>
            {chart.data && chart.variables === f.n && (
              <div className="mt-2 text-sm">
                {chart.data.is_chart ? <ChunkText text={chart.data.markdown} maxRows={60} /> : <p className="text-dim">The model says this is not a chart or table.</p>}
                <p className="mt-1 text-xs text-warn">{chart.data.warning} ({chart.data.model}, {chart.data.tokens} tokens)</p>
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function PageSearch({ docId }: { docId: string }) {
  const qc = useQueryClient();
  const [query, setQuery] = useState("");
  const [building, setBuilding] = useState(false);
  const st = useQuery({ queryKey: ["page-idx", docId], queryFn: () => api<PageIdx>(`/documents/${docId}/page-images`),
    refetchInterval: (q) => (building && !(q.state.data && q.state.data.indexed_pages >= (q.state.data.pages ?? 1)) ? 2500 : false) });
  const build = useMutation({ mutationFn: () => api(`/documents/${docId}/page-images`, { method: "POST" }), onSuccess: () => { setBuilding(true); void qc.invalidateQueries({ queryKey: ["page-idx", docId] }); } });
  const search = useMutation({ mutationFn: () => api<{ results: PageHit[]; note: string }>(`/documents/${docId}/page-images/search`, { method: "POST", json: { query, top_k: 6 } }) });
  const ready = !!st.data && st.data.indexed_pages > 0 && st.data.indexed_pages >= (st.data.pages ?? 0);
  return (
    <div className="space-y-3">
      <p className="text-sm text-dim">A local CLIP model looks at every page and at your words, in the same space, so you can find a page by what it looks like. It matches layout and large text, not small print.</p>
      {st.data && !st.data.available && <p className="text-sm text-warn">{st.data.detail}</p>}
      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={() => build.mutate()} disabled={build.isPending || !st.data?.available || (building && !ready)}>
          {ready ? "Rebuild the page index" : building ? `Indexing… ${st.data?.indexed_pages ?? 0}/${st.data?.pages ?? "?"}` : "Build the page index"}
        </Button>
        <span className="text-xs text-dim">{st.data ? `${st.data.indexed_pages} of ${st.data.pages ?? "?"} pages indexed. The first build downloads about 350 MB of models.` : ""}</span>
      </div>
      <ErrorBox error={build.error ?? search.error} />
      <form className="flex flex-wrap items-end gap-3" onSubmit={(e) => { e.preventDefault(); search.mutate(); }}>
        <div className="min-w-64 flex-1"><Field label="Describe the page"><Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. a flowchart of the review process" /></Field></div>
        <Button type="submit" disabled={!query.trim() || !ready || search.isPending}>Find pages</Button>
      </form>
      <ul className="grid grid-cols-2 gap-3 md:grid-cols-3">
        {search.data?.results.map((r) => (
          <li key={r.page} className="rounded-lg border border-border bg-surface p-2">
            <AuthImg path={r.image_url} alt={`Page ${r.page}`} className="w-full rounded border border-border" />
            <div className="mt-1 flex justify-between text-xs"><span className="font-medium">Page {r.page}</span><span className="font-mono text-dim">{r.score.toFixed(3)}</span></div>
          </li>
        ))}
      </ul>
      {search.data && <p className="text-xs text-dim">{search.data.note}</p>}
    </div>
  );
}

export default function VisualLab() {
  const { project, loading } = useProject();
  const pid = project?.id;
  const docs = useQuery({ queryKey: ["docs-ready", pid], enabled: !!pid, queryFn: () => api<{ items: Doc[] }>(`/projects/${pid}/documents?status=ready&limit=500`) });
  const pdfs = (docs.data?.items ?? []).filter((d) => d.filename.toLowerCase().endsWith(".pdf"));
  const [docState, setDoc] = useState<string | null>(null);
  const [tab, setTab] = useState<"crops" | "pages">("crops");
  const docId = docState ?? pdfs[0]?.id ?? "";
  const doc = pdfs.find((d) => d.id === docId);

  if (loading || docs.isLoading) return <Spinner />;
  if (!project) return <Empty title="No project selected"><Link className="text-accent underline" href="/projects">Create a project first</Link></Empty>;
  if (pdfs.length === 0) return <Empty title="No PDF yet">Upload a PDF in <Link className="text-accent underline" href="/ingestion">Document Ingestion</Link> first.</Empty>;
  return (
    <div className="w-full min-w-0">
      <PageHeader title="Visual Lab" subtitle="Work with what is not text: cut figures out of pages, have a vision model read a chart into a table, and find pages by how they look." />
      <div className="mb-4 grid gap-3 rounded-lg border border-border bg-surface p-3 md:grid-cols-[1fr_auto]">
        <Field label="Document"><Select value={docId} onChange={(e) => setDoc(e.target.value)} aria-label="Document">{pdfs.map((d) => <option key={d.id} value={d.id}>{d.filename}</option>)}</Select></Field>
        <div role="tablist" aria-label="Tool" className="flex items-end gap-1">
          {([["crops", "Figures and charts"], ["pages", "Find pages by appearance"]] as const).map(([id, label]) => (
            <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}
              className={clsx("rounded-md border px-3 py-1.5 text-sm", tab === id ? "border-accent bg-accent-soft text-accent" : "border-border bg-surface hover:bg-surface-2")}>{label}</button>
          ))}
        </div>
      </div>
      {docId && (tab === "crops" ? <Crops key={docId} docId={docId} pages={doc?.latest_version?.pages ?? 1} /> : <PageSearch key={docId} docId={docId} />)}
    </div>
  );
}
