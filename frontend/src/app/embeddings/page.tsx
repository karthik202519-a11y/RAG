"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { Construction } from "@/components/construction";
import { useMemo, useState } from "react";
import { ProjectionPlot } from "@/components/lazy-charts";
import { SERIES_COLORS } from "@/components/charts-colors";
import { Badge, Button, Card, Empty, ErrorBox, Field, Input, PageHeader, Select, Spinner } from "@/components/ui";
import { api, type ExploreResponse, type IndexInfo, type SimilarityResponse } from "@/lib/api";
import { useProject } from "@/lib/project";

const COLORS = SERIES_COLORS;

type Pt = ExploreResponse["points"][number];

export default function EmbeddingExplorer() {
  const { project, loading } = useProject();
  const pid = project?.id;
  const indexes = useQuery({ queryKey: ["indexes", pid], enabled: !!pid, queryFn: () => api<IndexInfo[]>(`/projects/${pid}/indexes`) });
  const ready = (indexes.data ?? []).filter((i) => i.status === "ready");
  const [idxState, setIdx] = useState<string | null>(null);
  const indexId = idxState ?? ready[0]?.id ?? "";
  const [sample, setSample] = useState(300);
  const [seed, setSeed] = useState(1);
  const [picked, setPicked] = useState<Pt[]>([]);
  const [query, setQuery] = useState("");

  const explore = useQuery({
    queryKey: ["explore", indexId, sample, seed], enabled: !!indexId,
    queryFn: () => api<ExploreResponse>(`/indexes/${indexId}/explore?sample=${sample}&seed=${seed}`),
  });
  const files = useMemo(() => [...new Set((explore.data?.points ?? []).map((p) => p.filename))], [explore.data]);
  const sim = useMutation({ mutationFn: () => api<SimilarityResponse>(`/indexes/${indexId}/similarity`, { method: "POST", json: { query, chunk_ids: picked.map((p) => p.chunk_id) } }) });
  const matrix = useMutation({ mutationFn: () => api<{ chunks: { label: string; preview: string }[]; matrix: number[][] }>(`/indexes/${indexId}/similarity-matrix`, { method: "POST", json: { chunk_ids: picked.map((p) => p.chunk_id) } }) });

  const toggle = (p: Pt) => {
    setPicked((cur) => (cur.some((x) => x.chunk_id === p.chunk_id) ? cur.filter((x) => x.chunk_id !== p.chunk_id) : cur.length >= 12 ? cur : [...cur, p]));
    sim.reset(); matrix.reset();
  };

  if (loading || indexes.isLoading) return <Spinner />;
  if (!project) return <Empty title="No project selected"><Link className="text-accent underline" href="/projects">Create a project first</Link></Empty>;
  if (ready.length === 0) return <Empty title="No ready index">Build one in <Link className="text-accent underline" href="/indexes">Indexes &amp; Embeddings</Link> first.</Empty>;
  const e = explore.data;
  const idx = ready.find((i) => i.id === indexId);

  return (
    <div className="w-full min-w-0">
      <PageHeader title="Embedding Explorer"
        subtitle="Each chunk is stored as one vector. Here is what those vectors look like, and how different similarity measures score them. Nothing here is a made-up illustration: it is read from your index." />
      <Construction indexId={indexId} kind="embeddings" />
      <div className="mb-4 grid gap-3 rounded-lg border border-border bg-surface p-3 md:grid-cols-[1fr_auto_auto_auto]">
        <Field label="Index"><Select value={indexId} onChange={(ev) => { setIdx(ev.target.value); setPicked([]); }} aria-label="Index">
          {ready.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}</Select></Field>
        <Field label="Chunks to plot"><Input type="number" min={3} max={2000} className="w-28" value={sample} onChange={(ev) => setSample(Number(ev.target.value))} /></Field>
        <Field label="Random seed"><Input type="number" className="w-24" value={seed} onChange={(ev) => setSeed(Number(ev.target.value))} /></Field>
        <div className="flex items-end text-xs text-dim">{idx && <span>{idx.provider} / {idx.model}<br />{idx.dimension} dimensions</span>}</div>
      </div>
      <ErrorBox error={explore.error} />

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_21rem]">
        <Card title={e ? `${e.sampled} of ${e.total_chunks} chunks, projected from ${e.dimension} dimensions to 2` : "Projection"}>
          {explore.isLoading ? <Spinner label="Projecting" /> : e && (
            <>
              <p className="mb-2 rounded bg-warn-soft px-2 py-1 text-xs text-warn">{e.warning}</p>
              <div className="h-[min(55vh,28rem)]" aria-label="Scatter plot of chunk vectors projected to two dimensions">
                <ProjectionPlot points={e.points} files={files} explained={e.explained_variance[0]} selected={picked.map((x) => x.chunk_id)} onToggle={toggle} />
              </div>
              <div className="mt-2 flex flex-wrap gap-3 text-xs">{files.map((f, i) => <span key={f} className="flex items-center gap-1"><span className="size-2.5 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />{f}</span>)}</div>
              <p className="mt-2 text-xs text-dim">Click points to select up to 12 chunks, then compare them on the right. Colour = document.</p>
            </>
          )}
        </Card>

        <div className="space-y-4">
          <Card title={`Selected chunks (${picked.length}/12)`} actions={picked.length > 0 ? <Button size="sm" variant="ghost" onClick={() => { setPicked([]); sim.reset(); matrix.reset(); }}>Clear</Button> : undefined}>
            {picked.length === 0 ? <p className="text-sm text-dim">Click points in the plot.</p> : (
              <ul className="space-y-1 text-xs">{picked.map((p) => <li key={p.chunk_id} className="truncate"><span className="font-mono">{p.filename} #{p.chunk_index}</span> <span className="text-dim">{p.preview}</span></li>)}</ul>
            )}
            <div className="mt-3 space-y-2">
              <Field label="Compare against a question"><Input value={query} onChange={(ev) => setQuery(ev.target.value)} placeholder="e.g. Which rocket carried Apollo?" /></Field>
              <div className="flex gap-2">
                <Button variant="primary" size="sm" disabled={!query.trim() || picked.length === 0} loading={sim.isPending} onClick={() => sim.mutate()}>Score with 3 measures</Button>
                <Button size="sm" disabled={picked.length < 2} loading={matrix.isPending} onClick={() => matrix.mutate()}>Similarity matrix</Button>
              </div>
              <ErrorBox error={sim.error ?? matrix.error} />
            </div>
          </Card>
          {sim.data && (
            <Card title="Cosine vs dot product vs Euclidean" pad={false}>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-border text-dim"><tr><th className="px-3 py-1.5">Chunk</th><th>cosine</th><th>dot</th><th>euclid.</th><th>norm</th></tr></thead>
                  <tbody className="divide-y divide-border font-mono">
                    {sim.data.rows.map((r) => (
                      <tr key={r.chunk_id}><td className="max-w-[9rem] truncate px-3 py-1.5 font-sans" title={r.preview}>{r.filename} #{r.chunk_index}</td>
                        <td>{r.cosine} <span className="text-dim">#{r.rank_cosine}</span></td><td>{r.dot} <span className="text-dim">#{r.rank_dot}</span></td>
                        <td>{r.euclidean} <span className="text-dim">#{r.rank_euclidean}</span></td><td>{r.norm}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="px-3 py-2 text-xs text-dim">Query vector length {sim.data.query_norm}. {sim.data.note}</p>
            </Card>
          )}
          {matrix.data && (
            <Card title="Pairwise cosine similarity">
              <div className="overflow-x-auto">
                <table className="text-xs" aria-label="Similarity matrix">
                  <tbody>
                    {matrix.data.matrix.map((row, i) => (
                      <tr key={i}><th className="max-w-[7rem] truncate pr-2 text-left font-normal text-dim" title={matrix.data.chunks[i].preview}>{matrix.data.chunks[i].label}</th>
                        {row.map((v, j) => <td key={j} title={`${matrix.data!.chunks[i].label} vs ${matrix.data!.chunks[j].label}`}
                          className={clsx("size-9 text-center font-mono", i === j && "font-bold")}
                          style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(Math.max(0, v) * 100)}%, transparent)` }}>{v.toFixed(2)}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-xs text-dim">Darker = more similar. The diagonal is each chunk against itself (1.00).</p>
            </Card>
          )}
          <Card><Badge tone="neutral">How to read this</Badge>
            <p className="mt-2 text-xs text-dim">One vector is stored per chunk, not per word or token. Chunks about the same topic usually land near each other, but a 2D picture keeps only a slice of the geometry, so use the score tables, not the plot, for any decision.</p></Card>
        </div>
      </div>
    </div>
  );
}
