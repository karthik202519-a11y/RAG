"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Download, FlaskConical, Plus } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, Button, Card, Empty, ErrorBox, Field, Input, PageHeader, Select, Spinner, statusTone } from "@/components/ui";
import { API_URL, api, getToken, type Doc, type EvalCompare, type EvalDatasetDetail, type EvalDatasetInfo, type EvalRunInfo, type IndexInfo, type MetricAgg } from "@/lib/api";
import { useProject } from "@/lib/project";

const METRIC_LABELS: Record<string, string> = {
  hit: "Hit rate", recall: "Recall", precision: "Precision", mrr: "MRR", ndcg: "nDCG", ap: "MAP", keyword_coverage: "Keyword coverage",
};
const HELP: Record<string, string> = {
  hit: "Share of questions with a labelled page in the top k.", recall: "Share of the labelled evidence items found in the top k (matters for multi-hop).",
  precision: "Share of the top k that is relevant.", mrr: "Mean of 1 / rank of the first relevant result.",
  ndcg: "Rewards putting relevant chunks early. Binary relevance; ideal = one chunk per evidence item.", ap: "Average precision at each relevant hit.",
  keyword_coverage: "Share of expected keywords present in the top-k text. A rough proxy, not answer quality.",
};
const pct = (v: number | null | undefined) => (v == null ? "not available" : v.toFixed(3));
const ci = (a?: Partial<MetricAgg>) => (a?.ci95 ? `${a.ci95[0].toFixed(2)}–${a.ci95[1].toFixed(2)}` : "no interval");

export default function EvaluationCenter() {
  const qc = useQueryClient();
  const { project, loading } = useProject();
  const pid = project?.id;
  const [dsState, setDs] = useState<string | null>(null);
  const [name, setName] = useState("");
  const datasets = useQuery({ queryKey: ["eval-datasets", pid], enabled: !!pid, queryFn: () => api<EvalDatasetInfo[]>(`/projects/${pid}/eval-datasets`) });
  const dsId = dsState ?? datasets.data?.[0]?.id ?? "";
  const create = useMutation({
    mutationFn: () => api<EvalDatasetInfo>(`/projects/${pid}/eval-datasets`, { method: "POST", json: { name } }),
    onSuccess: (d) => { setName(""); setDs(d.id); void qc.invalidateQueries({ queryKey: ["eval-datasets", pid] }); },
  });
  if (loading) return <Spinner />;
  if (!project) return <Empty title="No project selected"><Link className="text-accent underline" href="/projects">Create a project first</Link></Empty>;

  return (
    <div className="mx-auto max-w-[1500px]">
      <PageHeader title="Evaluation Center"
        subtitle="Label questions with where the answer lives, run them against an index, and compare configurations on the same questions. Metrics that need evidence you have not provided show 'not available', never zero." />
      <div className="grid gap-5 xl:grid-cols-[18rem_minmax(0,1fr)]">
        <div className="space-y-4">
          <Card title="Datasets">
            <ul className="mb-3 space-y-1">
              {datasets.data?.map((d) => (
                <li key={d.id}><button onClick={() => setDs(d.id)} className={clsx("w-full rounded-md px-2.5 py-1.5 text-left text-sm hover:bg-surface-2", d.id === dsId && "bg-accent-soft font-medium")}>
                  {d.name}<span className="block text-xs font-normal text-dim">{d.questions} questions · {d.runs} runs</span></button></li>
              ))}
              {datasets.data?.length === 0 && <li className="text-sm text-dim">None yet.</li>}
            </ul>
            <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
              <Input aria-label="New dataset name" placeholder="New dataset" value={name} onChange={(e) => setName(e.target.value)} required />
              <Button type="submit" loading={create.isPending}><Plus className="size-4" aria-hidden /></Button>
            </form>
            <ErrorBox error={create.error} />
          </Card>
        </div>
        {dsId ? <DatasetView key={dsId} id={dsId} pid={pid!} /> : <Empty title="Create a dataset to begin">Then add questions or import the earlier prototype&apos;s question files.</Empty>}
      </div>
    </div>
  );
}

function DatasetView({ id, pid }: { id: string; pid: string }) {
  const qc = useQueryClient();
  const [tab, setTab] = useState<"questions" | "runs" | "compare">("runs");
  const [picked, setPicked] = useState<string[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const ds = useQuery({ queryKey: ["eval-ds", id], queryFn: () => api<EvalDatasetDetail>(`/eval-datasets/${id}`) });
  const runs = useQuery({
    queryKey: ["eval-runs", id], queryFn: () => api<EvalRunInfo[]>(`/eval-datasets/${id}/runs`),
    refetchInterval: (q) => (q.state.data?.some((r) => r.status === "queued" || r.status === "running") ? 1500 : false),
  });
  if (ds.isLoading) return <Spinner />;
  const d = ds.data;
  if (!d) return <ErrorBox error={ds.error} />;
  const togglePick = (rid: string) => setPicked((p) => (p.includes(rid) ? p.filter((x) => x !== rid) : [...p, rid]));

  return (
    <div className="min-w-0 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div><h2 className="text-lg font-semibold">{d.name}</h2>
          <p className="text-xs text-dim">{d.questions.length} questions ({d.answerable} answerable, {d.unanswerable} unanswerable) · splits {Object.entries(d.splits).map(([k, v]) => `${k} ${v}`).join(", ") || "none"}</p></div>
        <div className="flex gap-1 rounded-md bg-surface-2 p-1 text-sm" role="tablist" aria-label="Dataset sections">
          {(["questions", "runs", "compare"] as const).map((t) => (
            <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)} className={clsx("rounded px-3 py-1 capitalize", tab === t ? "bg-surface font-medium shadow-sm" : "text-dim")}>
              {t}{t === "compare" && picked.length > 0 ? ` (${picked.length})` : ""}</button>
          ))}
        </div>
      </div>
      {d.runs_on_test_split > 0 && <p className="rounded bg-warn-soft px-3 py-2 text-xs text-warn">{d.guidance} The test split has been used in {d.runs_on_test_split} run(s) so far.</p>}

      {tab === "questions" && <Questions d={d} pid={pid} onChange={() => { void qc.invalidateQueries({ queryKey: ["eval-ds", id] }); void qc.invalidateQueries({ queryKey: ["eval-datasets", pid] }); }} />}
      {tab === "runs" && (
        <>
          <RunForm id={id} pid={pid} onStarted={() => { void qc.invalidateQueries({ queryKey: ["eval-runs", id] }); void qc.invalidateQueries({ queryKey: ["eval-ds", id] }); }} disabled={d.questions.length === 0} />
          <RunsTable runs={runs.data ?? []} picked={picked} onPick={togglePick} onOpen={setOpen} onCompare={() => setTab("compare")} />
          {open && <RunDetail id={open} onClose={() => setOpen(null)} />}
        </>
      )}
      {tab === "compare" && <Compare dsId={id} picked={picked} runs={runs.data ?? []} onPick={togglePick} />}
    </div>
  );
}

function Questions({ d, pid, onChange }: { d: EvalDatasetDetail; pid: string; onChange: () => void }) {
  const docs = useQuery({ queryKey: ["docs-ready", pid], queryFn: () => api<{ items: Doc[] }>(`/projects/${pid}/documents?status=ready&limit=500`) });
  const [form, setForm] = useState({ question: "", qtype: "", answerable: true, document: "", pages: "", keywords: "" });
  const [imp, setImp] = useState({ text: "", format: "auto", doc: "" });
  const [importMsg, setImportMsg] = useState<string | null>(null);
  const add = useMutation({
    mutationFn: () => api(`/eval-datasets/${d.id}/questions`, { method: "POST", json: {
      question: form.question, qtype: form.qtype, answerable: form.answerable,
      evidence: form.answerable ? [{ document: form.document, pages: form.pages.split(",").map((x) => parseInt(x, 10)).filter((n) => !Number.isNaN(n)) }] : [],
      keywords: form.keywords.split(",").map((x) => x.trim()).filter(Boolean) } }),
    onSuccess: () => { setForm({ ...form, question: "", pages: "", keywords: "" }); onChange(); },
  });
  const doImport = useMutation({
    mutationFn: () => api<{ added: number; skipped: { question: string; reason: string }[]; format: string }>(`/eval-datasets/${d.id}/import`, {
      method: "POST", json: { questions: JSON.parse(imp.text), format: imp.format, default_document: imp.doc } }),
    onSuccess: (r) => { setImportMsg(`Imported ${r.added} (${r.format})${r.skipped.length ? `, skipped ${r.skipped.length}: ${r.skipped[0].reason}` : ""}`); setImp({ ...imp, text: "" }); onChange(); },
  });
  const split = useMutation({ mutationFn: () => api(`/eval-datasets/${d.id}/assign-splits`, { method: "POST", json: { train: 0.6, validation: 0.2, seed: 1 } }), onSuccess: onChange });
  const del = useMutation({ mutationFn: (qid: string) => api(`/eval-questions/${qid}`, { method: "DELETE" }), onSuccess: onChange });
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_22rem]">
      <Card title={`Questions (${d.questions.length})`} pad={false} actions={<Button size="sm" onClick={() => split.mutate()} loading={split.isPending} title="60% train, 20% validation, 20% test, seeded">Assign splits</Button>}>
        {d.questions.length === 0 ? <div className="p-4"><Empty title="No questions yet">Add one, or import a JSON list.</Empty></div> : (
          <div className="max-h-[60vh] overflow-auto"><table className="w-full text-left text-sm">
            <thead className="sticky top-0 border-b border-border bg-surface text-xs text-dim"><tr><th className="px-3 py-2">Question</th><th>Type</th><th>Evidence</th><th>Split</th><th /></tr></thead>
            <tbody className="divide-y divide-border">{d.questions.map((q) => (
              <tr key={q.id}><td className="max-w-md px-3 py-1.5">{q.question}{!q.answerable && <Badge tone="warn">unanswerable</Badge>}</td><td className="text-xs text-dim">{q.qtype}</td>
                <td className="text-xs font-mono">{q.evidence.map((e) => `${e.document || "any"} p${e.pages.join(",") || "*"}`).join(" + ") || "-"}</td>
                <td><Badge>{q.split}</Badge></td><td><button className="px-2 text-xs text-bad hover:underline" onClick={() => del.mutate(q.id)} aria-label={`Delete question ${q.position + 1}`}>delete</button></td></tr>
            ))}</tbody></table></div>)}
      </Card>
      <div className="space-y-4">
        <Card title="Add a question">
          <form className="space-y-2" onSubmit={(e) => { e.preventDefault(); add.mutate(); }}>
            <Field label="Question"><textarea value={form.question} onChange={(e) => setForm({ ...form, question: e.target.value })} rows={2} required aria-label="Question text" className="w-full rounded-md border border-border bg-surface p-2 text-sm" /></Field>
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={form.answerable} onChange={(e) => setForm({ ...form, answerable: e.target.checked })} /> The documents contain the answer</label>
            {form.answerable && <>
              <Field label="Document" hint="Where the answer is. Retrieval counts a chunk as relevant when it comes from this document and these pages.">
                <Select value={form.document} onChange={(e) => setForm({ ...form, document: e.target.value })} aria-label="Evidence document">
                  <option value="">Any document</option>{docs.data?.items.map((x) => <option key={x.id} value={x.filename}>{x.filename}</option>)}</Select></Field>
              <Field label="Pages (comma separated)"><Input value={form.pages} onChange={(e) => setForm({ ...form, pages: e.target.value })} placeholder="171, 172" aria-label="Evidence pages" /></Field>
              <Field label="Expected keywords (optional)"><Input value={form.keywords} onChange={(e) => setForm({ ...form, keywords: e.target.value })} /></Field></>}
            <Field label="Type (optional)"><Input value={form.qtype} onChange={(e) => setForm({ ...form, qtype: e.target.value })} placeholder="prose, table, diagram…" /></Field>
            <ErrorBox error={add.error} />
            <Button type="submit" variant="primary" loading={add.isPending}>Add question</Button>
          </form>
        </Card>
        <Card title="Import questions (JSON)">
          <div className="space-y-2">
            <textarea value={imp.text} onChange={(e) => setImp({ ...imp, text: e.target.value })} rows={4} spellCheck={false} aria-label="Questions JSON" placeholder='[{"question": "...", "evidence": [{"document": "nasa", "pages": [171]}]}]' className="w-full rounded-md border border-border bg-surface p-2 font-mono text-xs" />
            <div className="grid grid-cols-2 gap-2">
              <Field label="Format"><Select value={imp.format} onChange={(e) => setImp({ ...imp, format: e.target.value })} aria-label="Import format"><option value="auto">Detect</option><option value="native">Studio</option><option value="prototype_basic">Prototype: basic</option><option value="prototype_hard">Prototype: hard set</option></Select></Field>
              <Field label="Default document" hint="For formats without one"><Input value={imp.doc} onChange={(e) => setImp({ ...imp, doc: e.target.value })} placeholder="nasa" /></Field>
            </div>
            <ErrorBox error={doImport.error} />{importMsg && <p className="text-xs text-good">{importMsg}</p>}
            <Button onClick={() => { setImportMsg(null); doImport.mutate(); }} loading={doImport.isPending} disabled={!imp.text.trim()}>Import</Button>
          </div>
        </Card>
      </div>
    </div>
  );
}

function RunForm({ id, pid, onStarted, disabled }: { id: string; pid: string; onStarted: () => void; disabled: boolean }) {
  const indexes = useQuery({ queryKey: ["indexes", pid], queryFn: () => api<IndexInfo[]>(`/projects/${pid}/indexes`) });
  const ready = (indexes.data ?? []).filter((i) => i.status === "ready");
  const [f, setF] = useState({ index: "", mode: "dense", k: 5, split: "all", transform: "none", rerank: "none", mmr: false, answers: false, label: "" });
  const index = f.index || ready[0]?.id || "";
  const start = useMutation({
    mutationFn: () => api(`/eval-datasets/${id}/runs`, { method: "POST", json: { index_id: index, mode: f.mode, k: f.k, split: f.split, transform: f.transform,
      rerank: f.rerank, mmr: f.mmr, with_answers: f.answers, label: f.label } }),
    onSuccess: () => { setF({ ...f, label: "" }); onStarted(); },
  });
  const llm = f.transform !== "none" || f.rerank !== "none" || f.answers;
  return (
    <Card title="Run an evaluation">
      {ready.length === 0 ? <p className="text-sm text-dim">Build an index first in <Link className="text-accent underline" href="/indexes">Indexes &amp; Embeddings</Link>.</p> : (
        <form className="grid gap-3 md:grid-cols-4" onSubmit={(e) => { e.preventDefault(); start.mutate(); }}>
          <Field label="Index"><Select value={index} onChange={(e) => setF({ ...f, index: e.target.value })} aria-label="Index to evaluate">{ready.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}</Select></Field>
          <Field label="Retrieval"><Select value={f.mode} onChange={(e) => setF({ ...f, mode: e.target.value })} aria-label="Retrieval mode"><option value="dense">Dense</option><option value="bm25">BM25</option><option value="hybrid">Hybrid</option></Select></Field>
          <Field label="Top k"><Input type="number" min={1} max={50} value={f.k} onChange={(e) => setF({ ...f, k: Number(e.target.value) })} aria-label="k" /></Field>
          <Field label="Split" hint="Tune on train/validation; use test once."><Select value={f.split} onChange={(e) => setF({ ...f, split: e.target.value })} aria-label="Split"><option value="all">All</option><option value="train">Train</option><option value="validation">Validation</option><option value="test">Test</option></Select></Field>
          <Field label="Query transform"><Select value={f.transform} onChange={(e) => setF({ ...f, transform: e.target.value })} aria-label="Query transform"><option value="none">None</option><option value="rewrite">Rewrite</option><option value="multi_query">Multi-query</option><option value="hyde">HyDE</option><option value="decompose">Decompose</option></Select></Field>
          <Field label="Reranking"><Select value={f.rerank} onChange={(e) => setF({ ...f, rerank: e.target.value })} aria-label="Reranking"><option value="none">None</option><option value="llm">LLM</option></Select></Field>
          <Field label="Label (optional)"><Input value={f.label} onChange={(e) => setF({ ...f, label: e.target.value })} placeholder="e.g. hybrid + rerank" /></Field>
          <div className="flex flex-col justify-end gap-1">
            <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={f.mmr} onChange={(e) => setF({ ...f, mmr: e.target.checked })} /> MMR diversity</label>
            <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={f.answers} onChange={(e) => setF({ ...f, answers: e.target.checked })} /> Also generate answers</label>
          </div>
          <div className="md:col-span-4">
            {llm && <p className="mb-2 text-xs text-warn">These options call the chat model for every question and spend tokens.</p>}
            <ErrorBox error={start.error} />
            <Button type="submit" variant="primary" loading={start.isPending} disabled={disabled}><FlaskConical className="size-4" aria-hidden /> Start run</Button>
            {disabled && <span className="ml-3 text-xs text-dim">Add questions first.</span>}
          </div>
        </form>)}
    </Card>
  );
}

function configLabel(r: EvalRunInfo) {
  const c = r.config;
  return [c.mode, `k=${c.k}`, c.split !== "all" ? c.split : "", c.transform !== "none" ? c.transform : "", c.rerank !== "none" ? `rerank:${c.rerank}` : "", c.mmr ? "mmr" : ""].filter(Boolean).join(" · ");
}

function RunsTable({ runs, picked, onPick, onOpen, onCompare }: { runs: EvalRunInfo[]; picked: string[]; onPick: (id: string) => void; onOpen: (id: string) => void; onCompare: () => void }) {
  return (
    <Card title={`Runs (${runs.length})`} pad={false} actions={<Button size="sm" disabled={picked.length < 2} onClick={onCompare}>Compare {picked.length || ""} selected</Button>}>
      {runs.length === 0 ? <p className="p-4 text-sm text-dim">No runs yet.</p> : (
        <div className="overflow-x-auto"><table className="w-full text-left text-sm">
          <thead className="border-b border-border text-xs text-dim"><tr><th className="px-3 py-2" /><th>Run</th><th>Status</th><th>Hit</th><th>MRR</th><th>nDCG</th><th>Recall</th><th>p50 ms</th></tr></thead>
          <tbody className="divide-y divide-border">{runs.map((r) => {
            const m = r.metrics?.retrieval;
            return (<tr key={r.id} className="hover:bg-surface-2">
              <td className="px-3"><input type="checkbox" checked={picked.includes(r.id)} disabled={r.status !== "done"} onChange={() => onPick(r.id)} aria-label={`Select run ${r.label}`} /></td>
              <td className="py-2"><button className="text-left font-medium hover:underline" onClick={() => onOpen(r.id)}>{r.label}</button><div className="text-xs text-dim">{configLabel(r)}</div></td>
              <td><Badge tone={statusTone(r.status === "done" ? "succeeded" : r.status)}>{r.status}</Badge>{r.status === "running" && r.job && <span className="ml-1 text-xs text-dim">{Math.round(r.job.progress * 100)}%</span>}</td>
              {(["hit", "mrr", "ndcg", "recall"] as const).map((k) => <td key={k} className="font-mono">{m ? pct(m[k]?.mean) : "-"}</td>)}
              <td className="font-mono">{r.metrics?.latency_ms.p50 != null ? Math.round(r.metrics.latency_ms.p50) : "-"}</td></tr>);
          })}</tbody></table></div>)}
    </Card>
  );
}

function RunDetail({ id, onClose }: { id: string; onClose: () => void }) {
  const q = useQuery({ queryKey: ["eval-run", id], queryFn: () => api<EvalRunInfo>(`/eval-runs/${id}`), refetchInterval: (x) => (x.state.data?.status === "running" || x.state.data?.status === "queued" ? 1500 : false) });
  const [onlyMisses, setOnlyMisses] = useState(false);
  const r = q.data;
  if (!r) return <Card><Spinner /></Card>;
  const m = r.metrics;
  const download = async (fmt: "json" | "csv") => {
    const res = await fetch(`${API_URL}/eval-runs/${id}/export?format=${fmt}`, { headers: { Authorization: `Bearer ${getToken()}` } });
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a"); a.href = url; a.download = `eval-${id}.${fmt}`; a.click(); URL.revokeObjectURL(url);
  };
  return (
    <Card title={r.label} actions={<><Button size="sm" onClick={() => download("csv")}><Download className="size-3.5" aria-hidden /> CSV</Button><Button size="sm" onClick={() => download("json")}>JSON</Button><Button size="sm" variant="ghost" onClick={onClose}>Close</Button></>}>
      <ErrorBox error={r.error ? new Error(r.error) : null} />
      {r.status !== "done" ? <Spinner label={`Running ${r.job?.stage ?? ""}`} /> : m && (
        <div className="space-y-5">
          <p className="text-xs text-dim">{m.answerable} answerable question(s) scored at k={m.k}{m.unanswerable ? `, ${m.unanswerable} unanswerable (not scored for retrieval)` : ""}{m.failed_questions ? ` · ${m.failed_questions} failed` : ""} · {r.seconds}s · {m.llm_tokens} LLM tokens</p>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="Retrieval metrics">
            {Object.entries(m.retrieval).map(([k, v]) => (
              <div key={k} className="rounded-lg border border-border px-3 py-2" title={HELP[k]}>
                <div className="text-xs text-dim">{METRIC_LABELS[k] ?? k}</div>
                <div className="font-mono text-lg font-semibold">{v.mean == null ? <span className="text-sm text-dim">not available</span> : v.mean.toFixed(3)}</div>
                <div className="text-[11px] text-dim">n={v.n} · 95% CI {v.ci95 ? `${v.ci95[0].toFixed(2)}–${v.ci95[1].toFixed(2)}` : "needs ≥10"}</div>
              </div>))}
          </div>
          {Object.keys(m.by_type).length > 0 && (
            <div><h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">By question type</h3>
              <table className="text-sm"><thead className="text-xs text-dim"><tr><th className="pr-4 text-left">Type</th><th className="pr-4">n</th><th className="pr-4">hit</th><th className="pr-4">MRR</th><th className="pr-4">nDCG</th></tr></thead>
                <tbody className="font-mono">{Object.entries(m.by_type).map(([t, v]) => <tr key={t}><td className="pr-4 font-sans">{t}</td><td className="pr-4">{v.n}</td><td className="pr-4">{pct(v.hit)}</td><td className="pr-4">{pct(v.mrr)}</td><td className="pr-4">{pct(v.ndcg)}</td></tr>)}</tbody></table></div>)}
          {m.unanswerable_gate.sweep && (
            <div><h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">Refusing off-topic questions with a score gate</h3>
              <p className="mb-1 text-xs text-dim">Best dense score: unanswerable {m.unanswerable_gate.top1_dense_unanswerable?.min.toFixed(2)}–{m.unanswerable_gate.top1_dense_unanswerable?.max.toFixed(2)}; answerable min {m.unanswerable_gate.top1_dense_answerable?.min.toFixed(2)}, median {m.unanswerable_gate.top1_dense_answerable?.median.toFixed(2)}. A threshold trades answerable questions kept against unanswerable ones refused:</p>
              <table className="text-xs"><thead className="text-dim"><tr><th className="pr-4 text-left">min score</th><th className="pr-4">answerable kept</th><th>unanswerable refused</th></tr></thead>
                <tbody className="font-mono">{m.unanswerable_gate.sweep.map((s) => <tr key={s.threshold}><td className="pr-4">{s.threshold}</td><td className="pr-4">{(s.answerable_kept * 100).toFixed(0)}%</td><td>{(s.unanswerable_refused * 100).toFixed(0)}%</td></tr>)}</tbody></table></div>)}
          {m.answers && (
            <div><h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">Generated answers ({m.answers.n})</h3>
              <ul className="text-sm"><li>Correct answer-or-refuse decisions: {pct(m.answers.abstention_correct.rate)} (n={m.answers.abstention_correct.n})</li>
                <li>Citations that resolve: {pct(m.answers.citations_resolve.rate)} (n={m.answers.citations_resolve.n})</li><li>Expected keywords present: {pct(m.answers.keyword_hit.rate)} (n={m.answers.keyword_hit.n})</li></ul>
              <p className="mt-1 text-xs text-dim">Not available: {m.answers.not_available.join("; ")}.</p></div>)}
          <div>
            <div className="mb-1 flex items-center justify-between"><h3 className="text-xs font-semibold uppercase tracking-wide text-dim">Per question</h3>
              <label className="flex items-center gap-1.5 text-xs"><input type="checkbox" checked={onlyMisses} onChange={(e) => setOnlyMisses(e.target.checked)} /> only misses</label></div>
            <div className="max-h-80 overflow-auto"><table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-surface text-dim"><tr><th className="py-1 pr-2">Question</th><th>Rank</th><th>Top result</th></tr></thead>
              <tbody className="divide-y divide-border">{(r.per_question ?? []).filter((p) => !onlyMisses || (p.answerable && !p.metrics.first_rank)).map((p) => (
                <tr key={p.question_id}><td className="max-w-md py-1 pr-2">{p.question}{!p.answerable && <Badge tone="warn">unanswerable</Badge>}{p.error && <span className="text-bad"> {p.error}</span>}</td>
                  <td className={clsx("font-mono", p.answerable && !p.metrics.first_rank && "text-bad")}>{p.answerable ? (p.metrics.first_rank ?? "miss") : "n/a"}</td>
                  <td className="font-mono text-dim">{p.top[0] ? `${p.top[0].filename} p${p.top[0].pages.join(",")}` : "-"}</td></tr>))}</tbody></table></div>
          </div>
          <ul className="list-disc space-y-0.5 pl-5 text-[11px] text-dim">{m.notes.map((n) => <li key={n}>{n}</li>)}</ul>
        </div>)}
    </Card>
  );
}

function Compare({ dsId, picked, runs, onPick }: { dsId: string; picked: string[]; runs: EvalRunInfo[]; onPick: (id: string) => void }) {
  const cmp = useQuery({ queryKey: ["eval-compare", dsId, picked.join(",")], enabled: picked.length >= 2, queryFn: () => api<EvalCompare>(`/eval-datasets/${dsId}/compare?runs=${picked.join(",")}`) });
  const done = runs.filter((r) => r.status === "done");
  const c = cmp.data;
  const chart = c ? ["hit", "mrr", "ndcg", "recall", "precision"].map((m) => ({ metric: METRIC_LABELS[m], ...Object.fromEntries(c.runs.map((r, i) => [r.label, c.metrics.find((x) => x.metric === m)?.values[i]?.mean ?? 0])) })) : [];
  const colors = ["#0f766e", "#b45309", "#4f46e5", "#be185d", "#2563eb"];
  return (
    <div className="space-y-4">
      <Card title="Choose runs (the first selected is the baseline)">
        {done.length < 2 ? <p className="text-sm text-dim">Finish at least two runs on this dataset.</p> : (
          <ul className="grid gap-1 md:grid-cols-2">{done.map((r) => <li key={r.id}><label className="flex items-start gap-2 text-sm"><input type="checkbox" className="mt-1" checked={picked.includes(r.id)} onChange={() => onPick(r.id)} />
            <span>{r.label}<span className="block text-xs text-dim">{configLabel(r)}</span></span></label></li>)}</ul>)}
      </Card>
      <ErrorBox error={cmp.error} />
      {c && (<>
        <Card title="Metrics side by side" pad={false}>
          <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="border-b border-border text-xs text-dim"><tr><th className="px-3 py-2">Metric</th>{c.runs.map((r) => <th key={r.id} className="px-2">{r.label}</th>)}</tr></thead>
            <tbody className="divide-y divide-border">{c.metrics.map((row) => (
              <tr key={row.metric}><td className="px-3 py-1.5" title={HELP[row.metric]}>{METRIC_LABELS[row.metric] ?? row.metric}</td>
                {row.values.map((v, i) => <td key={i} className="px-2 font-mono">{pct(v.mean)} <span className="text-[11px] text-dim">[{ci(v)}]</span></td>)}</tr>))}</tbody></table></div>
        </Card>
        <Card title="Chart"><div className="h-64" aria-label="Metrics for each run"><ResponsiveContainer width="100%" height="100%"><BarChart data={chart}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" /><XAxis dataKey="metric" tick={{ fontSize: 11, fill: "var(--text-dim)" }} /><YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "var(--text-dim)" }} />
          <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12 }} /><Legend wrapperStyle={{ fontSize: 12 }} />
          {c.runs.map((r, i) => <Bar key={r.id} dataKey={r.label} fill={colors[i % colors.length]} />)}</BarChart></ResponsiveContainer></div></Card>
        {c.paired_vs_first.map((p) => {
          const other = c.runs.find((r) => r.id === p.other);
          return (
            <Card key={p.other} title={`${other?.label} vs ${c.runs[0].label} (paired on ${p.shared_questions} questions)`}>
              <ul className="mb-3 grid gap-1 text-sm md:grid-cols-2">{(["hit", "mrr", "ndcg", "recall"] as const).map((m) => {
                const d = p[m] as { mean_diff: number; ci95: [number, number] | null; significant: boolean | null } | null;
                return <li key={m}><b>{METRIC_LABELS[m]}</b>: {d ? <span className="font-mono">{d.mean_diff >= 0 ? "+" : ""}{d.mean_diff.toFixed(3)}</span> : "n/a"}{" "}
                  {d && (d.significant === null ? <span className="text-xs text-dim">(too few questions for an interval)</span> : d.significant ? <Badge tone="good">interval excludes 0</Badge> : <Badge>not distinguishable from noise</Badge>)}</li>;
              })}</ul>
              <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">Questions whose first relevant rank changed ({p.changed_questions.length})</h4>
              <ul className="max-h-56 space-y-1 overflow-auto text-xs">{p.changed_questions.map((q) => (
                <li key={q.question_id}><Badge tone={q.direction === "improved" ? "good" : "bad"}>{q.direction}</Badge> <span className="font-mono">{q.baseline_rank ?? "miss"} → {q.other_rank ?? "miss"}</span> {q.question}</li>))}</ul>
            </Card>);
        })}
        <p className="text-xs text-dim">{c.note}</p>
      </>)}
    </div>
  );
}
