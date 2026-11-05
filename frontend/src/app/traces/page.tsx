"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { useState } from "react";
import { Badge, Card, Empty, ErrorBox, PageHeader, Spinner, Stat } from "@/components/ui";
import { api } from "@/lib/api";
import { useProject } from "@/lib/project";

type Trace = {
  id: string; kind: "search" | "answer" | "job"; name: string; detail: string; at: string; total_ms: number;
  spans: { name: string; ms: number }[]; attrs: Record<string, string | number | boolean | null>; results: number | null;
};
type Traces = { items: Trace[]; summary: { count: number; searches: number; answers: number; jobs: number; failed_jobs: number; p50_ms: number | null; p95_ms: number | null; tokens: number } };

const KINDS = [{ id: "", label: "All" }, { id: "search", label: "Searches" }, { id: "answer", label: "Answers" }, { id: "job", label: "Jobs" }];
const ms = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${Math.round(n)} ms`);

/** Spans drawn left to right in the order recorded; widths are proportional to the longest trace shown. */
function Waterfall({ t, scale }: { t: Trace; scale: number }) {
  const starts = t.spans.map((_, i) => t.spans.slice(0, i).reduce((a, x) => a + x.ms, 0));
  return (
    <ol className="space-y-1" aria-label="Steps">
      {t.spans.map((s, i) => {
        const left = (starts[i] / scale) * 100;
        return (
          <li key={i} className="grid grid-cols-[8.5rem_1fr_4.5rem] items-center gap-2 text-xs">
            <span className="truncate text-dim">{s.name}</span>
            <span className="relative h-3 rounded bg-surface-2">
              <span className="absolute inset-y-0 rounded bg-accent" style={{ left: `${left}%`, width: `${Math.max((s.ms / scale) * 100, 0.8)}%` }} />
            </span>
            <span className="text-right font-mono">{ms(s.ms)}</span>
          </li>
        );
      })}
    </ol>
  );
}

export default function TracesPage() {
  const { project, loading } = useProject();
  const pid = project?.id;
  const [kind, setKind] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const q = useQuery({
    queryKey: ["traces", pid, kind], enabled: !!pid, refetchInterval: 10_000,
    queryFn: () => api<Traces>(`/projects/${pid}/traces?limit=100${kind ? `&kind=${kind}` : ""}`),
  });

  if (loading) return <Spinner />;
  if (!project) return <Empty title="No project selected"><Link className="text-accent underline" href="/projects">Create a project first</Link></Empty>;
  const d = q.data;
  const scale = Math.max(1, ...(d?.items ?? []).map((t) => t.total_ms));

  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title="Traces" subtitle="Every search, answer and background job in this project, with the steps the code timed while it ran. Times are measured, not estimated. Refreshes every 10 seconds." />
      {d && (
        <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Recorded" value={d.summary.count} sub={`${d.summary.searches} searches · ${d.summary.answers} answers · ${d.summary.jobs} jobs`} />
          <Stat label="Median query" value={d.summary.p50_ms != null ? ms(d.summary.p50_ms) : "n/a"} sub="searches and answers" />
          <Stat label="Slowest 5%" value={d.summary.p95_ms != null ? ms(d.summary.p95_ms) : "n/a"} sub="p95" />
          <Stat label="Failed jobs" value={d.summary.failed_jobs} sub={`${d.summary.tokens.toLocaleString()} answer tokens`} />
        </div>
      )}
      <div role="tablist" aria-label="Filter" className="mb-3 flex gap-1">
        {KINDS.map((k) => (
          <button key={k.id} role="tab" aria-selected={kind === k.id} onClick={() => setKind(k.id)}
            className={clsx("rounded-md border px-2.5 py-1 text-sm", kind === k.id ? "border-accent bg-accent-soft text-accent" : "border-border bg-surface hover:bg-surface-2")}>{k.label}</button>
        ))}
      </div>
      <ErrorBox error={q.error} />
      {q.isLoading && <Spinner />}
      {d && d.items.length === 0 && <Empty title="Nothing recorded yet">Run a search in the <Link className="text-accent underline" href="/retrieval">Retrieval Lab</Link> and it will appear here.</Empty>}
      <ul className="space-y-2">
        {d?.items.map((t) => (
          <li key={t.id}>
            <Card pad={false}>
              <button className="flex w-full items-center gap-3 px-4 py-2.5 text-left" aria-expanded={open === t.id} onClick={() => setOpen(open === t.id ? null : t.id)}>
                <Badge tone={t.kind === "job" ? (t.attrs.status === "failed" || t.attrs.status === "dead" ? "bad" : "neutral") : t.kind === "answer" ? "accent" : "good"}>{t.kind}</Badge>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium">{t.name}</span>
                  <span className="block truncate text-xs text-dim">{t.detail} · {new Date(t.at).toLocaleString()}</span>
                </span>
                <span className="font-mono text-sm tabular-nums">{t.total_ms ? ms(t.total_ms) : "n/a"}</span>
              </button>
              {open === t.id && (
                <div className="space-y-3 border-t border-border px-4 py-3">
                  {t.spans.length > 0 ? <Waterfall t={t} scale={scale} /> : <p className="text-xs text-dim">No timing was recorded for this item.</p>}
                  <dl className="flex flex-wrap gap-x-6 gap-y-1 text-xs">
                    {Object.entries(t.attrs).map(([k, v]) => <div key={k}><dt className="inline text-dim">{k.replace(/_/g, " ")}: </dt><dd className="inline font-mono">{String(v)}</dd></div>)}
                    {t.results != null && <div><dt className="inline text-dim">passages: </dt><dd className="inline font-mono">{t.results}</dd></div>}
                  </dl>
                </div>
              )}
            </Card>
          </li>
        ))}
      </ul>
    </div>
  );
}
