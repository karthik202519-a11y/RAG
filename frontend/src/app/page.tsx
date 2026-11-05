"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { Badge, Card, Empty, ErrorBox, PageHeader, Spinner, Stat, fmt, statusTone } from "@/components/ui";
import { api, type EvalDatasetInfo, type IndexInfo, type IngestionStats, type Job } from "@/lib/api";
import { useProject } from "@/lib/project";

const JOB_LABEL: Record<string, string> = {
  run_eval: "Evaluation run", build_index: "Index build", build_graph: "Graph build", embed_pages: "Visual index", describe_figures: "Figure descriptions", ingest_document: "Parse document",
};

export default function Dashboard() {
  const { project, loading } = useProject();
  const pid = project?.id;
  const stats = useQuery({ queryKey: ["stats", pid], enabled: !!pid, refetchInterval: 4000, queryFn: () => api<IngestionStats>(`/projects/${pid}/ingestion-stats`) });
  const jobs = useQuery({ queryKey: ["jobs-recent", pid], enabled: !!pid, refetchInterval: 4000, queryFn: () => api<{ items: Job[] }>(`/projects/${pid}/ingestion-jobs?limit=8`) });
  const indexes = useQuery({ queryKey: ["indexes", pid], enabled: !!pid, queryFn: () => api<IndexInfo[]>(`/projects/${pid}/indexes`) });
  const datasets = useQuery({ queryKey: ["eval-datasets", pid], enabled: !!pid, queryFn: () => api<EvalDatasetInfo[]>(`/projects/${pid}/eval-datasets`) });

  if (loading) return <Spinner />;
  if (!project) {
    return <Empty title="Welcome to RAG">Create a project to begin. <Link className="text-accent underline" href="/projects">Go to Projects</Link></Empty>;
  }
  const s = stats.data;
  const ready = (indexes.data ?? []).filter((i) => i.status === "ready").length;
  const runs = (datasets.data ?? []).reduce((a, d) => a + d.runs, 0);
  const next = !s?.documents ? { href: "/ingestion", text: "Upload a document" } : !ready ? { href: "/indexes", text: "Build an index" }
    : { href: "/retrieval", text: "Ask a question" };

  const steps = [
    { href: "/ingestion", label: "Documents", n: s?.documents, sub: s ? `${fmt.n(s.pages)} pages · ${fmt.n(s.tokens)} tokens` : "" },
    { href: "/indexes", label: "Indexes", n: indexes.data?.length, sub: `${ready} ready` },
    { href: "/retrieval", label: "Retrieval", n: undefined, sub: "ask and compare" },
    { href: "/evaluation", label: "Evaluation", n: datasets.data?.length, sub: `${runs} run${runs === 1 ? "" : "s"}` },
  ];

  return (
    <div className="w-full min-w-0 space-y-5">
      <PageHeader title={project.name} subtitle={project.description || "Where this project stands, from documents to measured results."}
        actions={<Link href={next.href} className="inline-flex h-9 items-center gap-2 rounded-md bg-accent px-3.5 text-sm font-medium text-accent-ink hover:brightness-110">{next.text}<ArrowRight className="size-4" aria-hidden /></Link>} />
      <ErrorBox error={stats.error ?? indexes.error ?? datasets.error ?? jobs.error} />
      <section aria-labelledby="start-heading" className="border-y border-border py-5">
        <h2 id="start-heading" className="text-lg font-semibold">Build understanding, one step at a time</h2>
        <p className="mt-1 max-w-3xl text-sm text-dim">Upload a source, inspect what was extracted, compare chunks, then build an index. Search the evidence before asking for an answer. Use evaluation to compare your choices.</p>
        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-sm">
          <Link href="/parsing" className="text-accent underline underline-offset-4">Inspect extraction</Link>
          <Link href="/chunking" className="text-accent underline underline-offset-4">Compare chunking methods</Link>
          <Link href="/recommendations" className="text-accent underline underline-offset-4">Get suggestions for this project</Link>
          <Link href="/settings" className="text-accent underline underline-offset-4">Check provider connections</Link>
        </div>
      </section>

      <ol className="grid gap-3 md:grid-cols-4" aria-label="Pipeline progress">
        {steps.map((st, i) => (
          <li key={st.label}>
            <Link href={st.href} className="block h-full rounded-lg border border-border bg-surface px-4 py-3 hover:bg-surface-2">
              <div className="flex items-center justify-between text-xs text-dim"><span>{st.label}</span>{i < steps.length - 1 && <ArrowRight className="size-3.5 max-md:hidden" aria-hidden />}</div>
              <div className="mt-1 font-mono text-2xl font-semibold tabular-nums">{st.n ?? "·"}</div>
              <div className="text-xs text-dim">{st.sub}</div>
            </Link>
          </li>
        ))}
      </ol>

      <div className="grid gap-5 lg:grid-cols-3">
        <Card title="Recent activity" className="lg:col-span-2" pad={false} actions={<Link href="/ingestion" className="text-xs text-accent underline">Open ingestion</Link>}>
          {jobs.data?.items.length ? (
            <ul className="divide-y divide-border">
              {jobs.data.items.map((j) => (
                <li key={j.id} className="flex items-center justify-between gap-3 px-4 py-2 text-sm">
                  <span className="min-w-0 truncate"><span className="text-dim">{JOB_LABEL[j.kind] ?? j.kind}</span>{j.filename ? ` · ${j.filename}` : ""}</span>
                  <span className="flex shrink-0 items-center gap-2">
                    {j.status === "running" && <span className="font-mono text-xs text-dim">{Math.round(j.progress * 100)}%</span>}
                    <Badge tone={statusTone(j.status)}>{j.status}</Badge>
                  </span>
                </li>
              ))}
            </ul>
          ) : <p className="p-4 text-sm text-dim">Nothing has run yet. <Link className="text-accent underline" href="/ingestion">Upload a document</Link> to start.</p>}
        </Card>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-1">
          <Stat label="Queue depth" value={fmt.n(s?.queue_depth)} sub={s?.extraction_success_rate != null ? `${fmt.pct(s.extraction_success_rate)} of parses succeeded` : "no finished parses yet"} />
          <Stat label="Tables found" value={fmt.n(s?.tables)} sub={s ? `${fmt.bytes(s.storage_bytes)} stored` : undefined} />
        </div>
      </div>
    </div>
  );
}
