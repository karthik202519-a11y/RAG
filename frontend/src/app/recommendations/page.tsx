"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { Badge, Card, Empty, ErrorBox, PageHeader, Spinner, Stat } from "@/components/ui";
import { api } from "@/lib/api";
import { useProject } from "@/lib/project";

type Stage = {
  id: string; title: string; basis: "measured" | "rule" | "default"; reasons: string[]; evidence: { label: string; value: string | number }[];
  pick: { method: string; label: string; params: Record<string, unknown> }; alternatives: { method: string; why: string }[];
  action: { href: string; label: string } | null;
};
type Recs = { documents: number; indexes: number; runs: number; questions: number; tokens: number; stages: Stage[]; next_experiments: string[] };

const BASIS = {
  measured: { tone: "good" as const, label: "Measured", hint: "Backed by your own evaluation runs" },
  rule: { tone: "accent" as const, label: "Rule", hint: "From measurements of your documents" },
  default: { tone: "neutral" as const, label: "Starting point", hint: "Not enough information yet" },
};

export default function Recommendations() {
  const { project, loading } = useProject();
  const pid = project?.id;
  const q = useQuery({ queryKey: ["recs", pid], enabled: !!pid, queryFn: () => api<Recs>(`/projects/${pid}/recommendations`), staleTime: 30_000 });

  if (loading || q.isLoading) return <Spinner />;
  if (!project) return <Empty title="No project selected"><Link className="text-accent underline" href="/projects">Create a project first</Link></Empty>;
  const d = q.data;
  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <PageHeader title="Suggestions" subtitle="A pick for every stage of the pipeline, with the reasons and the numbers behind it. A pick marked Measured comes from your own evaluation runs and overrides the rules; the rest come from measurements of your documents. Nothing here calls a model." />
      <ErrorBox error={q.error} />
      {d && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Documents" value={d.documents} sub={`${d.tokens.toLocaleString()} tokens`} />
            <Stat label="Indexes" value={d.indexes} />
            <Stat label="Evaluation questions" value={d.questions} />
            <Stat label="Completed runs" value={d.runs} />
          </div>
          <ol className="space-y-3">
            {d.stages.map((s) => (
              <li key={s.id}>
                <Card title={<span className="flex flex-wrap items-center gap-2">{s.title}<Badge tone={BASIS[s.basis].tone} title={BASIS[s.basis].hint}>{BASIS[s.basis].label}</Badge></span>}
                  actions={s.action ? <Link href={s.action.href} className="text-sm text-accent underline">{s.action.label}</Link> : undefined}>
                  <p className="text-base font-semibold">{s.pick.label}</p>
                  {Object.keys(s.pick.params).length > 0 && <p className="mt-0.5 font-mono text-xs text-dim">{Object.entries(s.pick.params).map(([k, v]) => `${k}=${String(v)}`).join("  ")}</p>}
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">{s.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
                  {s.evidence.length > 0 && (
                    <details className="mt-3 text-sm">
                      <summary className="cursor-pointer text-dim hover:text-text">Evidence ({s.evidence.length})</summary>
                      <dl className="mt-2 space-y-1">{s.evidence.map((e, i) => <div key={i} className="grid gap-x-3 sm:grid-cols-[14rem_1fr]"><dt className="truncate text-dim" title={e.label}>{e.label}</dt><dd className="font-mono text-xs">{String(e.value)}</dd></div>)}</dl>
                    </details>
                  )}
                  {s.alternatives.length > 0 && <p className="mt-3 text-xs text-dim">Also consider: {s.alternatives.map((a) => `${a.method} (${a.why})`).join("; ")}.</p>}
                </Card>
              </li>
            ))}
          </ol>
          {d.next_experiments.length > 0 && (
            <Card title="Experiments worth running next">
              <ul className="list-disc space-y-1 pl-5 text-sm">{d.next_experiments.map((e, i) => <li key={i}>{e}</li>)}</ul>
              <p className="mt-2 text-xs text-dim">Run them together with the experiment grid in the <Link href="/evaluation" className="text-accent underline">Evaluation Center</Link>.</p>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
