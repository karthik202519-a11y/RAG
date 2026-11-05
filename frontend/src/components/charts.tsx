"use client";

// The only module that imports Recharts. Pages load it lazily (see lazy-charts.tsx), so the charting
// library never counts against a page's first load.
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { ExploreResponse } from "@/lib/api";

const TICK = { fontSize: 11, fill: "var(--text-dim)" };
const TIP = { background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12 };
import { SERIES_COLORS } from "./charts-colors";

export function Histogram({ bins }: { bins: { from: number; to: number; count: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={bins.map((h) => ({ range: `${h.from}–${h.to}`, count: h.count }))}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="range" tick={TICK} interval="preserveStartEnd" minTickGap={28} tickFormatter={(v: string) => v.split("–")[0]} />
        <YAxis allowDecimals={false} tick={TICK} />
        <Tooltip contentStyle={TIP} />
        <Bar dataKey="count" fill="var(--accent)" isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function RunsBarChart({ data, series }: { data: Record<string, string | number>[]; series: string[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="metric" tick={TICK} />
        <YAxis domain={[0, 1]} tick={TICK} />
        <Tooltip contentStyle={TIP} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((s, i) => <Bar key={s} dataKey={s} fill={SERIES_COLORS[i % SERIES_COLORS.length]} isAnimationActive={false} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}

type Pt = ExploreResponse["points"][number];

export function ProjectionPlot({ points, files, explained, selected, onToggle }: {
  points: Pt[]; files: string[]; explained: number; selected: string[]; onToggle: (p: Pt) => void;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis type="number" dataKey="x" name="PC1" tick={{ fontSize: 10, fill: "var(--text-dim)" }}
          label={{ value: `PC1 · ${(explained * 100).toFixed(1)}% of variance`, position: "insideBottom", offset: -2, fontSize: 11, fill: "var(--text-dim)" }} />
        <YAxis type="number" dataKey="y" name="PC2" tick={{ fontSize: 10, fill: "var(--text-dim)" }} />
        <ZAxis range={[40, 40]} />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} content={({ payload }) => {
          const p = payload?.[0]?.payload as Pt | undefined;
          return p ? <div className="max-w-xs rounded border border-border bg-surface p-2 text-xs shadow"><b>{p.filename} #{p.chunk_index}</b> · p.{p.pages.join(",")}<br />{p.preview}…</div> : null;
        }} />
        {files.map((f, i) => (
          <Scatter key={f} name={f} data={points.filter((p) => p.filename === f)} fill={SERIES_COLORS[i % SERIES_COLORS.length]} isAnimationActive={false}
            onClick={(d) => onToggle(d as unknown as Pt)}
            shape={(props: { cx?: number; cy?: number; payload?: Pt }) => {
              const sel = selected.includes(props.payload?.chunk_id ?? "");
              return <circle cx={props.cx} cy={props.cy} r={sel ? 7 : 4} fill={SERIES_COLORS[i % SERIES_COLORS.length]} stroke={sel ? "var(--text)" : "none"}
                strokeWidth={2} opacity={sel ? 1 : 0.75} style={{ cursor: "pointer" }} />;
            }} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
