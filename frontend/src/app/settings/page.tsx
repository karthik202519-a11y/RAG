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
  { target: "embedding", label: "Embedding deployment" }, { target: "chat", label: "Chat deployment" }, { target: "vision", label: "Vision deployment" },
];

export default function SettingsPage() {
  const q = useQuery({ queryKey: ["settings-overview"], queryFn: () => api<Overview>("/settings/overview"), staleTime: 30_000 });
  const [results, setResults] = useState<Record<string, TestResult>>({});
  const test = useMutation({
    mutationFn: (t: { target: string; model?: string }) => api<TestResult>("/settings/test", { method: "POST", json: t }),
    onSuccess: (r) => setResults((cur) => ({ ...cur, [r.target]: r })),
  });
  const running = test.isPending ? test.variables?.target : null;

  if (q.isLoading) return <Spinner />;
  const d = q.data;
  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <PageHeader title="Integrations & Settings" subtitle={`How this deployment is configured${d ? ` (version ${d.version})` : ""}. Secrets stay in the environment: this page shows whether each is set, never its value. Edit .env and restart the API to change one.`} />
      <ErrorBox error={q.error} />
      {d && (
        <>
          <Card title="Runtime">
            <dl className="grid gap-x-8 gap-y-2 text-sm sm:grid-cols-2">
              {d.runtime.map((r) => (
                <div key={r.name} className="flex items-baseline justify-between gap-3 border-b border-border pb-1.5">
                  <dt className="text-dim">{r.name}{r.env && <span className="ml-1 font-mono text-xs">({r.env})</span>}</dt>
                  <dd className="text-right font-medium">{r.value}</dd>
                </div>
              ))}
            </dl>
          </Card>
          <Card title="Azure OpenAI">
            <ul className="space-y-1.5 text-sm">
              {d.azure.map((a) => (
                <li key={a.env} className="flex items-center justify-between gap-3">
                  <span className="font-mono text-xs">{a.env}</span>
                  <span className="flex items-center gap-2">{a.value && <span className="text-xs text-dim">{a.value}</span>}<Badge tone={a.state === "set" ? "good" : "warn"}>{a.state}</Badge></span>
                </li>
              ))}
            </ul>
            <p className="mt-3 text-xs text-dim">Chat: {d.generation.configured ? `ready, models ${d.generation.models.join(", ") || "n/a"}` : d.generation.detail}. Vision: {d.vision.configured ? "ready" : d.vision.detail}.</p>
          </Card>
          <Card title="Connection tests">
            <p className="mb-3 text-xs text-dim">Each test makes one small real call and reports what happened. Embedding, chat and vision tests use a few tokens.</p>
            <ul className="divide-y divide-border">
              {TESTS.map((t) => {
                const r = results[t.target];
                return (
                  <li key={t.target} className="flex items-start gap-3 py-2">
                    <div className="w-44 shrink-0 text-sm font-medium">{t.label}</div>
                    <div className="min-w-0 flex-1 text-sm">
                      {r ? <span className={r.ok ? "text-good" : "text-bad"}>{r.ok ? "Working. " : "Failed. "}<span className="text-text">{r.detail}</span> <span className="text-xs text-dim">({r.ms} ms)</span></span> : <span className="text-dim">Not tested yet.</span>}
                    </div>
                    <Button size="sm" onClick={() => test.mutate({ target: t.target, model: t.model })} disabled={!!running}>{running === t.target ? "Testing…" : "Test"}</Button>
                  </li>
                );
              })}
            </ul>
          </Card>
          <ul className="space-y-1 text-xs text-dim">{d.notes.map((n) => <li key={n}>{n}</li>)}</ul>
        </>
      )}
    </div>
  );
}
