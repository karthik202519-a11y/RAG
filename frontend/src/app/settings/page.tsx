"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Badge, Button, Card, ErrorBox, PageHeader, Spinner } from "@/components/ui";
import { api } from "@/lib/api";

type Overview = {
  version: string;
  runtime: { name: string; value: string; env: string | null }[];
  azure: { env: string; state: "set" | "not set"; value?: string | null }[];
  embedding: { id: string; model: string; available: boolean; detail: string }[];
  generation: { configured: boolean; detail: string; models: string[] };
  vision: { configured: boolean; detail: string };
  notes: string[];
};
type TestResult = { target: string; ok: boolean; detail: string; ms: number };

const TESTS: { target: string; label: string; model?: string }[] = [
  { target: "database", label: "Database" }, { target: "storage", label: "Artifact storage" },
  { target: "qdrant", label: "Qdrant" }, { target: "faiss", label: "FAISS" },
  { target: "chroma", label: "Chroma" }, { target: "lancedb", label: "LanceDB" }, { target: "pgvector", label: "pgvector" },
  { target: "embedding", label: "Embedding deployment" }, { target: "chat", label: "Chat deployment" }, { target: "vision", label: "Vision deployment" },
];

export default function SettingsPage() {
  const q = useQuery({ queryKey: ["settings-overview"], queryFn: () => api<Overview>("/settings/overview"), staleTime: 30_000 });
  const [view,setView]=useState<"runtime"|"providers"|"connections">("runtime");
  const [results, setResults] = useState<Record<string, TestResult>>({});
  const test = useMutation({
    mutationFn: (t: { target: string; model?: string }) => api<TestResult>("/settings/test", { method: "POST", json: t }),
    onSuccess: (r) => setResults((cur) => ({ ...cur, [r.target]: r })),
  });
  const probe = useMutation({
    mutationFn: (id: string) => api<{ dimension: number; latency_ms: number }>(`/providers/${id}/test`, { method: "POST" }),
  });
  const running = test.isPending ? test.variables?.target : null;

  if (q.isLoading) return <Spinner />;
  const d = q.data;
  return (
    <div className="w-full min-w-0 space-y-4">
      <PageHeader title="Integrations & Settings" subtitle={`How this deployment is configured${d ? ` (version ${d.version})` : ""}. Secrets stay in the environment: this page shows whether each is set, never its value. Edit .env and restart the API to change one.`} />
      <ErrorBox error={q.error} />
      <div className="flex flex-wrap gap-2" role="group" aria-label="Settings view">{([{id:"runtime",label:"Runtime & configuration"},{id:"providers",label:"Embedding providers"},{id:"connections",label:"Connection tests"}] as const).map(v=><Button key={v.id} aria-pressed={view===v.id} variant={view===v.id?"primary":"secondary"} onClick={()=>setView(v.id)}>{v.label}</Button>)}</div>
      {d && (
        <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-2">
          {view==="runtime"&&<><Card title="Runtime">
            <dl className="grid gap-x-8 gap-y-2 text-sm">
              {d.runtime.map((r) => (
                <div key={r.name} className="flex min-w-0 flex-wrap items-baseline justify-between gap-3 border-b border-border pb-1.5">
                  <dt className="text-dim">{r.name}{r.env && <span className="ml-1 font-mono text-xs">({r.env})</span>}</dt>
                  <dd className="text-right font-medium">{r.value}</dd>
                </div>
              ))}
            </dl>
          </Card>
          <Card title="Azure OpenAI">
            <ul className="space-y-1.5 text-sm">
              {d.azure.map((a) => (
                <li key={a.env} className="flex min-w-0 flex-wrap items-center justify-between gap-3">
                  <span className="break-all font-mono text-xs">{a.env}</span>
                  <span className="flex min-w-0 flex-wrap items-center gap-2">{a.value && <span className="break-all text-xs text-dim">{a.value}</span>}<Badge tone={a.state === "set" ? "good" : "warn"}>{a.state}</Badge></span>
                </li>
              ))}
            </ul>
            <p className="mt-3 text-xs text-dim">Chat: {d.generation.configured ? `ready, models ${d.generation.models.join(", ") || "n/a"}` : d.generation.detail}. Vision: {d.vision.configured ? "ready" : d.vision.detail}.</p>
          </Card>
          </>}
          {view==="providers"&&<Card title="Embedding providers" className="xl:col-span-2">
            <p className="mb-3 text-sm text-dim">Choose a provider when building an index. Test a configured provider with one small embedding request; first use of a local model may download it.</p>
            <ul className="grid gap-x-8 divide-y divide-border lg:grid-cols-2">
              {d.embedding.map((p) => <li key={p.id} className="flex flex-wrap items-start gap-3 py-3">
                <div className="min-w-0 flex-1"><span className="font-medium">{p.id}</span><p className="break-words text-xs text-dim">{p.model} / {p.detail}</p>
                  {probe.variables === p.id && probe.data && <p role="status" className="text-xs text-good">Working: {probe.data.dimension} dimensions in {probe.data.latency_ms} ms.</p>}
                  {probe.variables === p.id && <ErrorBox error={probe.error} />}
                </div>
                <Badge tone={p.available ? "good" : "warn"}>{p.available ? "available" : "needs setup"}</Badge>
                <Button size="sm" aria-label={`Test ${p.id} embeddings`} disabled={!p.available || probe.isPending} loading={probe.isPending && probe.variables === p.id} onClick={() => probe.mutate(p.id)}>Test</Button>
              </li>)}
            </ul>
          </Card>
          }
          {view==="connections"&&<Card title="Connection tests" className="xl:col-span-2">
            <p className="mb-3 text-xs text-dim">Each test makes one small real call and reports what happened. Embedding, chat and vision tests use a few tokens.</p>
            <ul className="grid gap-x-8 divide-y divide-border lg:grid-cols-2">
              {TESTS.map((t) => {
                const r = results[t.target];
                return (
                  <li key={t.target} className="flex flex-wrap items-start gap-3 py-2 sm:flex-nowrap">
                    <div className="w-full text-sm font-medium sm:w-40 sm:shrink-0">{t.label}</div>
                    <div className="min-w-0 flex-1 text-sm">
                      {r ? <span className={r.ok ? "text-good" : "text-bad"}>{r.ok ? "Working. " : "Failed. "}<span className="text-text">{r.detail}</span> <span className="text-xs text-dim">({r.ms} ms)</span></span> : <span className="text-dim">Not tested yet.</span>}
                    </div>
                    <Button size="sm" onClick={() => test.mutate({ target: t.target, model: t.model })} disabled={!!running}>{running === t.target ? "Testing…" : "Test"}</Button>
                  </li>
                );
              })}
            </ul>
          </Card>
          }
          <ul className="space-y-1 text-xs text-dim xl:col-span-2">{d.notes.map((n) => <li key={n}>{n}</li>)}</ul>
        </div>
      )}
    </div>
  );
}
