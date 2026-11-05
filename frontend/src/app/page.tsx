"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { Badge, Card, Empty, ErrorBox, PageHeader, Spinner, Stat, fmt, statusTone } from "@/components/ui";
import { api, type Catalog, type IngestionStats, type Job } from "@/lib/api";
import { useProject } from "@/lib/project";

export default function Dashboard() {
  const { project, loading } = useProject();
  const stats = useQuery({
    queryKey: ["stats", project?.id], enabled: !!project, refetchInterval: 4000,
    queryFn: () => api<IngestionStats>(`/projects/${project!.id}/ingestion-stats`),
  });
  const jobs = useQuery({
    queryKey: ["jobs-recent", project?.id], enabled: !!project, refetchInterval: 4000,
    queryFn: () => api<{ items: Job[] }>(`/projects/${project!.id}/ingestion-jobs?limit=8`),
  });
  const cat = useQuery({ queryKey: ["catalog"], queryFn: () => api<Catalog>("/method-catalog") });

  if (loading) return <Spinner />;
  if (!project) {
    return <Empty title="Welcome to RAG Learning Studio">Create a project to begin. <Link className="text-accent underline" href="/projects">Go to Projects</Link></Empty>;
  }
  const s = stats.data;
  return (
    <div className="mx-auto max-w-6xl">
      <PageHeader title={project.name} subtitle={project.description || "Project dashboard"} />
      <ErrorBox error={stats.error} />
      <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Documents" value={fmt.n(s?.documents)} sub={s ? `${s.versions} versions` : undefined} />
        <Stat label="Pages" value={fmt.n(s?.pages)} sub={s ? `${fmt.n(s.tables)} tables` : undefined} />
        <Stat label="Tokens extracted" value={fmt.n(s?.tokens)} sub={s ? `${fmt.n(s.characters)} characters` : undefined} />
        <Stat label="Queue depth" value={fmt.n(s?.queue_depth)} sub={s?.extraction_success_rate != null ? `${fmt.pct(s.extraction_success_rate)} extraction success` : "no finished jobs yet"} />
      </div>
      <div className="grid gap-5 lg:grid-cols-3">
        <Card title="Recent jobs" className="lg:col-span-2" actions={<Link href="/ingestion" className="text-xs text-accent underline">Open ingestion</Link>}>
          {jobs.data?.items.length ? (
            <ul className="divide-y divide-border">
              {jobs.data.items.map((j) => (
                <li key={j.id} className="flex items-center justify-between gap-3 py-2">
                  <span className="truncate">{j.filename ?? j.kind}</span>
                  <span className="flex items-center gap-2">
                    {j.status === "running" && <span className="font-mono text-xs text-dim">{Math.round(j.progress * 100)}%</span>}
                    <Badge tone={statusTone(j.status)}>{j.status}</Badge>
                  </span>
                </li>
              ))}
            </ul>
          ) : <p className="text-sm text-dim">No jobs yet. <Link className="text-accent underline" href="/ingestion">Upload a document</Link> to start.</p>}
        </Card>
        <Card title="What works today">
          <ul className="space-y-1.5 text-sm">
            <li>Upload and parse PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, CSV, JSON and XML, with background jobs you can retry and cancel</li>
            <li>Chunking Playground: {cat.data?.items.filter((m) => m.category === "chunking" && m.status !== "unavailable").length ?? "…"} chunkers, compared side by side</li>
            <li>Embeddings into Qdrant, dense / BM25 / hybrid retrieval, query transforms, MMR and LLM reranking</li>
            <li>Grounded answers with checked citations; vision descriptions of diagram pages</li>
            <li>Evaluation Center: labelled questions, recall / MRR / nDCG with confidence intervals, paired run comparison</li>
            <li>{cat.data?.counts.implemented ?? "…"} methods run for real; {cat.data?.counts.unavailable ?? "…"} are described but not built</li>
          </ul>
          <p className="mt-3 text-xs text-dim">Not built: the parser comparison lab, GraphRAG, pipeline builder, experiment studio, tracing and settings screens (greyed in the sidebar).</p>
        </Card>
      </div>
    </div>
  );
}
