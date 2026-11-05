"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useMemo, useState } from "react";
import { SERIES_COLORS } from "@/components/charts-colors";
import { Badge, Button, Card, Empty, ErrorBox, Field, Input, PageHeader, Select, Spinner, Stat } from "@/components/ui";
import { api, type IndexInfo, type SearchResponse } from "@/lib/api";
import { useProject } from "@/lib/project";

type Node = { id: string; name: string; type: string; degree: number; mentions: number; community: number };
type Build = { status: string; model: string; chunks_requested: number; chunks_processed: number; chunks_failed: number; relations_dropped: number;
  tokens: number; error: string | null; seconds: number | null; progress: number | null; stage: string | null };
type GraphInfo = { build: Build | null; graph: null | { entities: number; relations: number; isolated: number;
  communities: { id: number; size: number; top: string[] }[]; nodes: Node[]; edges: { source: string; target: string; relation: string }[] } };
type GraphSearch = { seeds: { id: string; name: string }[]; note: string | null; edges: { source: string; relation: string; target: string }[];
  results: { rank: number; chunk_id: string; score: number; filename: string; pages: number[]; section: string | null; text: string; via: { source: string; relation: string; target: string }[] }[] };

const W = 760, H = 460;

/** A small force layout: repulsion between all nodes, springs along edges, gentle pull to the centre. Deterministic. */
function layout(nodes: Node[], edges: { source: string; target: string }[]) {
  const pos = new Map(nodes.map((n, i) => {
    const a = (i / Math.max(1, nodes.length)) * Math.PI * 2;
    return [n.id, { x: W / 2 + Math.cos(a) * 180, y: H / 2 + Math.sin(a) * 140, vx: 0, vy: 0 }];
  }));
  for (let it = 0; it < 220; it++) {
    const cool = 1 - it / 220;
    const list = [...pos.entries()];
    for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
      const a = list[i][1], b = list[j][1];
      let dx = a.x - b.x, dy = a.y - b.y;
      const d2 = Math.max(dx * dx + dy * dy, 25);
      const f = 2600 / d2, d = Math.sqrt(d2);
      dx /= d; dy /= d;
      a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }
    for (const e of edges) {
      const a = pos.get(e.source), b = pos.get(e.target);
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y, d = Math.max(Math.hypot(dx, dy), 1);
      const f = (d - 70) * 0.02;
      a.vx += (dx / d) * f; a.vy += (dy / d) * f; b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
    }
    for (const p of pos.values()) {
      p.vx += (W / 2 - p.x) * 0.004; p.vy += (H / 2 - p.y) * 0.004;
      p.x = Math.min(W - 20, Math.max(20, p.x + p.vx * 0.5 * cool)); p.y = Math.min(H - 20, Math.max(20, p.y + p.vy * 0.5 * cool));
      p.vx *= 0.6; p.vy *= 0.6;
    }
  }
  return pos;
}

function GraphCanvas({ graph }: { graph: NonNullable<GraphInfo["graph"]> }) {
  const [sel, setSel] = useState<string | null>(null);
  const pos = useMemo(() => layout(graph.nodes, graph.edges), [graph]);
  const nbrs = useMemo(() => {
    const m = new Map<string, Set<string>>();
    for (const e of graph.edges) { (m.get(e.source) ?? m.set(e.source, new Set()).get(e.source)!).add(e.target); (m.get(e.target) ?? m.set(e.target, new Set()).get(e.target)!).add(e.source); }
    return m;
  }, [graph]);
  const on = (id: string) => !sel || id === sel || nbrs.get(sel)?.has(id);
  const selected = graph.nodes.find((n) => n.id === sel);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-md border border-border bg-surface-2" role="img" aria-label={`Knowledge graph with ${graph.nodes.length} entities`}>
        {graph.edges.map((e, i) => {
          const a = pos.get(e.source), b = pos.get(e.target);
          if (!a || !b) return null;
          const hot = sel && (e.source === sel || e.target === sel);
          return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="currentColor" className="text-border" strokeOpacity={sel ? (hot ? 1 : 0.15) : 0.7} strokeWidth={hot ? 2 : 1}><title>{e.relation}</title></line>;
        })}
        {graph.nodes.map((n) => {
          const p = pos.get(n.id)!;
          const r = 4 + Math.min(10, Math.sqrt(n.degree) * 2.2);
          return (
            <g key={n.id} opacity={on(n.id) ? 1 : 0.2} className="cursor-pointer" onClick={() => setSel(sel === n.id ? null : n.id)}>
              <circle cx={p.x} cy={p.y} r={r} fill={SERIES_COLORS[(n.community < 0 ? 0 : n.community) % SERIES_COLORS.length]} stroke="var(--surface)" strokeWidth={1.5} />
              {(n.degree >= 3 || sel === n.id) && <text x={p.x + r + 3} y={p.y + 3} fontSize="10" fill="currentColor" className="text-text">{n.name.length > 24 ? n.name.slice(0, 23) + "…" : n.name}</text>}
              <title>{`${n.name}${n.type ? ` (${n.type})` : ""} · ${n.degree} relations · in ${n.mentions} chunks`}</title>
            </g>
          );
        })}
      </svg>
      <p className="mt-1 text-xs text-dim">{selected ? `${selected.name}${selected.type ? ` (${selected.type})` : ""}: ${selected.degree} relations, found in ${selected.mentions} chunks. Click again to clear.` : `Showing the ${graph.nodes.length} best-connected of ${graph.entities - graph.isolated} entities that have relations (${graph.isolated} have none). Colour is community, size is connections. Click a node to see its neighbours.`}</p>
    </div>
  );
}

export default function GraphPage() {
  const { project, loading } = useProject();
  const pid = project?.id;
  const qc = useQueryClient();
  const indexes = useQuery({ queryKey: ["indexes", pid], enabled: !!pid, queryFn: () => api<IndexInfo[]>(`/projects/${pid}/indexes`) });
  const ready = (indexes.data ?? []).filter((i) => i.status === "ready");
  const [idxState, setIdx] = useState<string | null>(null);
  const indexId = idxState ?? ready[0]?.id ?? "";
  const [maxChunks, setMaxChunks] = useState(150);
  const info = useQuery({
    queryKey: ["graph", indexId], enabled: !!indexId,
    queryFn: () => api<GraphInfo>(`/indexes/${indexId}/graph`),
    refetchInterval: (q) => (["queued", "running"].includes(q.state.data?.build?.status ?? "") ? 1500 : false),
  });
  const build = useMutation({
    mutationFn: () => api(`/indexes/${indexId}/graph`, { method: "POST", json: { max_chunks: maxChunks } }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["graph", indexId] }),
  });
  const [query, setQuery] = useState("");
  const [hops, setHops] = useState(1);
  const gsearch = useMutation({ mutationFn: () => api<GraphSearch>(`/indexes/${indexId}/graph/search`, { method: "POST", json: { query, top_k: 5, hops } }) });
  const dsearch = useMutation({ mutationFn: () => api<SearchResponse>(`/indexes/${indexId}/search`, { method: "POST", json: { query, mode: "dense", top_k: 5 } }) });

  if (loading || indexes.isLoading) return <Spinner />;
  if (!project) return <Empty title="No project selected"><Link className="text-accent underline" href="/projects">Create a project first</Link></Empty>;
  if (ready.length === 0) return <Empty title="No ready index">Build one in <Link className="text-accent underline" href="/indexes">Indexes &amp; Embeddings</Link> first.</Empty>;
  const b = info.data?.build, g = info.data?.graph;
  const busy = b?.status === "queued" || b?.status === "running";

  return (
    <div className="mx-auto max-w-[1400px]">
      <PageHeader title="GraphRAG Explorer" subtitle="A model reads chunks and lists the entities and relations they state. Anything it names that is not in the chunk text is thrown away, so every edge points at the chunk it came from. Then you can search by following relations and compare with plain vector search." />
      <div className="mb-4 grid gap-3 rounded-lg border border-border bg-surface p-3 md:grid-cols-[1fr_10rem_auto]">
        <Field label="Index"><Select value={indexId} onChange={(e) => setIdx(e.target.value)} aria-label="Index">{ready.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}</Select></Field>
        <Field label="Chunks to read" hint="One chat call each"><Input type="number" min={5} max={2000} value={maxChunks} onChange={(e) => setMaxChunks(Number(e.target.value))} /></Field>
        <div className="flex items-end"><Button onClick={() => build.mutate()} disabled={busy || build.isPending}>{busy ? "Building…" : b?.status === "ready" ? "Rebuild graph" : "Build graph"}</Button></div>
      </div>
      <ErrorBox error={build.error ?? info.error} />
      {busy && b && <div role="status" className="mb-4 rounded-md border border-border bg-surface px-3 py-2 text-sm">Reading chunks with {b.model}: {b.stage ?? "starting"}{b.progress != null && ` (${Math.round(b.progress * 100)}%)`}</div>}
      {b?.status === "failed" && <p className="mb-4 text-sm text-bad">The build failed: {b.error}</p>}
      {!b && !info.isLoading && <Empty title="No graph for this index yet">Choose how many chunks to read and press Build graph. Each chunk costs one chat-model call.</Empty>}

      {g && b && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Entities" value={g.entities} sub={`${g.isolated} with no relations`} />
            <Stat label="Relations" value={g.relations} sub={`${b.relations_dropped} rejected as not in the text`} />
            <Stat label="Chunks read" value={`${b.chunks_processed}/${b.chunks_requested}`} sub={b.chunks_failed ? `${b.chunks_failed} failed` : "none failed"} />
            <Stat label="Build" value={b.seconds != null ? `${b.seconds} s` : "n/a"} sub={`${b.tokens.toLocaleString()} tokens · ${b.model}`} />
          </div>
          <div className="grid gap-4 lg:grid-cols-[1fr_20rem]">
            <Card title="Graph"><GraphCanvas graph={g} /></Card>
            <Card title="Communities" pad>
              {g.communities.length === 0 ? <p className="text-sm text-dim">No relations, so no communities.</p> : (
                <ul className="space-y-2 text-sm">
                  {g.communities.map((c) => (
                    <li key={c.id} className="flex gap-2">
                      <span className="mt-1.5 size-2.5 shrink-0 rounded-full" style={{ background: SERIES_COLORS[c.id % SERIES_COLORS.length] }} aria-hidden />
                      <span><span className="font-medium">{c.size} entities</span><span className="block text-xs text-dim">{c.top.join(", ")}</span></span>
                    </li>
                  ))}
                </ul>
              )}
              <p className="mt-3 text-xs text-dim">Groups found by modularity. They are not summarised by a model.</p>
            </Card>
          </div>

          <Card title="Graph search vs vector search">
            <form className="mb-3 flex flex-wrap items-end gap-3" onSubmit={(e) => { e.preventDefault(); gsearch.mutate(); dsearch.mutate(); }}>
              <div className="min-w-64 flex-1"><Field label="Question"><Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Name an entity from the graph, e.g. the one you clicked above" /></Field></div>
              <Field label="Hops"><Select value={hops} onChange={(e) => setHops(Number(e.target.value))} aria-label="Hops"><option value={1}>1</option><option value={2}>2</option></Select></Field>
              <Button type="submit" disabled={!query.trim() || gsearch.isPending}>Search both</Button>
            </form>
            <ErrorBox error={gsearch.error ?? dsearch.error} />
            {(gsearch.data || dsearch.data) && (
              <div className="grid gap-4 lg:grid-cols-2">
                <div>
                  <h3 className="mb-1 text-sm font-semibold">Graph search</h3>
                  {gsearch.data?.seeds.length ? <p className="mb-2 text-xs text-dim">Started from: {gsearch.data.seeds.map((s) => s.name).join(", ")}</p> : null}
                  {gsearch.data?.note && <p className="text-sm text-warn">{gsearch.data.note}</p>}
                  <ol className="space-y-2">
                    {gsearch.data?.results.map((r) => (
                      <li key={r.chunk_id} className="rounded-md border border-border p-2 text-sm">
                        <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-dim"><Badge>#{r.rank}</Badge>{r.filename} p.{r.pages.join(", ")}<span>score {r.score}</span></div>
                        <p className="line-clamp-4 whitespace-pre-wrap">{r.text}</p>
                        <p className="mt-1 text-xs text-accent">{r.via.map((v) => `${v.source} → ${v.relation} → ${v.target}`).join(" · ")}</p>
                      </li>
                    ))}
                  </ol>
                </div>
                <div>
                  <h3 className="mb-1 text-sm font-semibold">Vector search (dense)</h3>
                  <ol className="space-y-2">
                    {dsearch.data?.results.map((r) => (
                      <li key={r.chunk_id} className="rounded-md border border-border p-2 text-sm">
                        <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-dim"><Badge>#{r.rank}</Badge>{r.citation}<span>score {r.score.toFixed(3)}</span></div>
                        <p className="line-clamp-4 whitespace-pre-wrap">{r.text}</p>
                      </li>
                    ))}
                  </ol>
                </div>
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
