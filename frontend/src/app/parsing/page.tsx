"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { useState } from "react";
import { Badge, Button, Card, Empty, ErrorBox, Field, Input, PageHeader, Select, Spinner } from "@/components/ui";
import { api, type Doc } from "@/lib/api";
import { useProject } from "@/lib/project";

type PageResult = { page: number; text: string; tables: number | null; seconds: number; chars: number; words: number; empty: boolean };
type Compare = {
  filename: string; pages: number[];
  results: { parser: string; error: string | null; seconds: number; pages: PageResult[] }[];
  agreement: { a: string; b: string; mean: number | null; per_page: number[] }[];
};
const PARSERS = [
  { id: "pymupdf", label: "PyMuPDF", note: "Used by ingestion" },
  { id: "pdfplumber", label: "pdfplumber", note: "Strong on ruled tables" },
  { id: "pypdf", label: "pypdf", note: "Pure Python, no tables" },
];

/** "1-3, 7" -> [1,2,3,7]; returns null when the text is not a page list. */
function parsePages(s: string): number[] | null {
  const out = new Set<number>();
  for (const part of s.split(",").map((x) => x.trim()).filter(Boolean)) {
    const m = /^(\d+)(?:\s*-\s*(\d+))?$/.exec(part);
    if (!m) return null;
    const a = Number(m[1]), b = Number(m[2] ?? m[1]);
    if (a < 1 || b < a || b - a > 50) return null;
    for (let p = a; p <= b; p++) out.add(p);
  }
  return out.size ? [...out].sort((x, y) => x - y) : null;
}

export default function ParserLab() {
  const { project, loading } = useProject();
  const pid = project?.id;
  const docs = useQuery({ queryKey: ["docs-ready", pid], enabled: !!pid, queryFn: () => api<{ items: Doc[] }>(`/projects/${pid}/documents?status=ready&limit=500`) });
  const pdfs = (docs.data?.items ?? []).filter((d) => d.filename.toLowerCase().endsWith(".pdf"));
  const [docState, setDoc] = useState<string | null>(null);
  const docId = docState ?? pdfs[0]?.id ?? "";
  const [pagesText, setPagesText] = useState("1-3");
  const [chosen, setChosen] = useState<string[]>(PARSERS.map((p) => p.id));
  const [view, setView] = useState(0);
  const pages = parsePages(pagesText);
  const run = useMutation({
    mutationFn: () => api<Compare>(`/documents/${docId}/parser-compare`, { method: "POST", json: { pages: (pages ?? []).slice(0, 12), parsers: chosen } }),
    onSuccess: () => setView(0),
  });

  if (loading || docs.isLoading) return <Spinner />;
  if (!project) return <Empty title="No project selected"><Link className="text-accent underline" href="/projects">Create a project first</Link></Empty>;
  if (pdfs.length === 0) return <Empty title="No PDF to compare">Upload a PDF in <Link className="text-accent underline" href="/ingestion">Document Ingestion</Link> first.</Empty>;
  const r = run.data;
  const pageNo = r?.pages[view];

  return (
    <div className="w-full min-w-0">
      <PageHeader title="Parser Lab" subtitle="Run several PDF parsers on the same pages and see what each one actually returned: text, tables found, time, and how much they agree. Up to 12 pages per run." />
      <div className="mb-4 grid gap-3 rounded-lg border border-border bg-surface p-3 md:grid-cols-[1fr_12rem_auto]">
        <Field label="Document"><Select value={docId} onChange={(e) => setDoc(e.target.value)} aria-label="Document">{pdfs.map((d) => <option key={d.id} value={d.id}>{d.filename}</option>)}</Select></Field>
        <Field label="Pages" hint={pages ? `${Math.min(pages.length, 12)} page${pages.length === 1 ? "" : "s"}` : "Use 1-3, 7"}><Input value={pagesText} onChange={(e) => setPagesText(e.target.value)} aria-invalid={!pages} /></Field>
        <div className="flex items-end"><Button onClick={() => run.mutate()} disabled={!pages || chosen.length === 0 || run.isPending}>{run.isPending ? "Comparing…" : "Compare parsers"}</Button></div>
        <fieldset className="flex flex-wrap gap-4 md:col-span-3">
          <legend className="sr-only">Parsers</legend>
          {PARSERS.map((p) => (
            <label key={p.id} className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={chosen.includes(p.id)} onChange={(e) => setChosen((c) => (e.target.checked ? [...c, p.id] : c.filter((x) => x !== p.id)))} />
              {p.label} <span className="text-xs text-dim">{p.note}</span>
            </label>
          ))}
        </fieldset>
      </div>
      <ErrorBox error={run.error} />
      {run.isPending && <Spinner label="Running parsers" />}
      {r && (
        <div className="space-y-4">
          <Card title="Summary">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs text-dim"><tr><th className="py-1 pr-4 font-medium">Parser</th><th className="pr-4 font-medium">Time</th><th className="pr-4 font-medium">Words</th><th className="pr-4 font-medium">Tables found</th><th className="pr-4 font-medium">Empty pages</th><th className="font-medium">Result</th></tr></thead>
                <tbody className="divide-y divide-border">
                  {r.results.map((x) => (
                    <tr key={x.parser}>
                      <td className="py-1.5 pr-4 font-medium">{x.parser}</td>
                      <td className="pr-4 tabular-nums">{(x.seconds * 1000).toFixed(0)} ms</td>
                      <td className="pr-4 tabular-nums">{x.pages.reduce((a, p) => a + p.words, 0).toLocaleString()}</td>
                      <td className="pr-4 tabular-nums">{x.pages[0]?.tables == null ? "not supported" : x.pages.reduce((a, p) => a + (p.tables ?? 0), 0)}</td>
                      <td className="pr-4 tabular-nums">{x.pages.filter((p) => p.empty).length}</td>
                      <td>{x.error ? <Badge tone="bad" title={x.error}>failed</Badge> : <Badge tone="good">ok</Badge>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {r.agreement.length > 0 && (
              <p className="mt-3 text-xs text-dim">
                Word-level agreement (1.0 means identical words in the same order):{" "}
                {r.agreement.map((a) => `${a.a} vs ${a.b} ${a.mean?.toFixed(2)}`).join(" · ")}. Low agreement means the parsers read the page differently; check the text below to see which is right.
              </p>
            )}
          </Card>
          <div role="tablist" aria-label="Page" className="flex flex-wrap gap-1">
            {r.pages.map((p, i) => (
              <button key={p} role="tab" aria-selected={i === view} onClick={() => setView(i)}
                className={clsx("rounded-md border px-2.5 py-1 text-sm", i === view ? "border-accent bg-accent-soft text-accent" : "border-border bg-surface hover:bg-surface-2")}>Page {p}</button>
            ))}
          </div>
          <div className="grid gap-3 lg:grid-cols-3">
            {r.results.map((x) => {
              const pg = x.pages.find((p) => p.page === pageNo);
              return (
                <Card key={x.parser} title={x.parser}>
                  {x.error ? <p className="text-sm text-bad">{x.error}</p> : !pg ? null : (
                    <>
                      <p className="mb-2 text-xs text-dim">{pg.words} words · {pg.tables == null ? "no table detection" : `${pg.tables} table${pg.tables === 1 ? "" : "s"}`} · {(pg.seconds * 1000).toFixed(0)} ms</p>
                      <pre className="max-h-[32rem] overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-surface-2 p-2 text-xs">{pg.empty ? "(no text extracted)" : pg.text}</pre>
                    </>
                  )}
                </Card>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
