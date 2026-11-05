"use client";

import clsx from "clsx";
import { memo, useMemo } from "react";
import type { Chunk } from "@/lib/api";

const RENDER_LIMIT = 120_000;

type Seg = { a: number; b: number; cover: number[]; startsOf: number[] };

/** Split the document at every chunk boundary and record which chunks cover each piece. */
function buildSegments(len: number, chunks: Chunk[]): Seg[] {
  const located = chunks.filter((c) => c.start != null && c.end != null && c.end! > c.start!);
  const pts = new Set<number>([0, len]);
  for (const c of located) { pts.add(Math.max(0, c.start!)); pts.add(Math.min(len, c.end!)); }
  const xs = [...pts].sort((x, y) => x - y);
  const byStart = new Map<number, number[]>();
  for (const c of located) { const l = byStart.get(c.start!) ?? []; l.push(c.index); byStart.set(c.start!, l); }
  const sorted = [...located].sort((x, y) => x.start! - y.start!);
  const active: Chunk[] = [];
  let next = 0;
  const segs: Seg[] = [];
  for (let i = 0; i < xs.length - 1; i++) {
    const a = xs[i], b = xs[i + 1];
    while (next < sorted.length && sorted[next].start! <= a) active.push(sorted[next++]);
    for (let k = active.length - 1; k >= 0; k--) if (active[k].end! <= a) active.splice(k, 1);
    segs.push({ a, b, cover: active.map((c) => c.index), startsOf: byStart.get(a) ?? [] });
  }
  return segs;
}

export const ChunkedText = memo(function ChunkedText({
  text, chunks, selected, onSelect,
}: { text: string; chunks: Chunk[]; selected: number | null; onSelect: (i: number) => void }) {
  const shown = text.length > RENDER_LIMIT ? text.slice(0, RENDER_LIMIT) : text;
  const segs = useMemo(() => buildSegments(shown.length, chunks), [shown.length, chunks]);
  return (
    <div>
      {text.length > RENDER_LIMIT && (
        <p className="mb-2 rounded bg-warn-soft px-2 py-1 text-xs text-warn">
          Showing the first {RENDER_LIMIT.toLocaleString()} of {text.length.toLocaleString()} characters. Narrow the page range to see the rest.
        </p>
      )}
      <div className="whitespace-pre-wrap break-words font-mono text-[12.5px] leading-6" data-testid="chunked-text">
        {segs.map((s, i) => {
          const piece = shown.slice(s.a, s.b);
          if (s.cover.length === 0) {
            return <span key={i} className="seg-gap" title="Not in any chunk">{piece}</span>;
          }
          const first = s.cover[0];
          const overlap = s.cover.length > 1;
          const isSel = selected != null && s.cover.includes(selected);
          return (
            <span key={i} role="button" tabIndex={0} aria-label={overlap ? `Chunks ${s.cover.map((c) => `#${c}`).join(" and ")} overlap here` : `Chunk #${first}`}
              aria-pressed={isSel}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); (e.currentTarget as HTMLElement).click(); } }}
              onClick={() => onSelect(selected != null && s.cover.includes(selected) && s.cover.length > 1
              ? s.cover[(s.cover.indexOf(selected) + 1) % s.cover.length] : first)}
              title={overlap ? `Overlap of chunks ${s.cover.map((c) => `#${c}`).join(" and ")}` : `Chunk #${first}`}
              style={{ background: `var(--c${first % 8})` }}
              className={clsx("seg", overlap && "seg-overlap", isSel && "seg-selected")}>
              {s.startsOf.length > 0 && (
                <sup className="mr-0.5 rounded bg-text px-1 text-[9px] font-semibold text-bg">#{s.startsOf[0]}</sup>
              )}
              {piece}
            </span>
          );
        })}
      </div>
    </div>
  );
});

export function ChunkDetail({ chunk, docText }: { chunk: Chunk; docText: string }) {
  const body = chunk.text ?? (chunk.start != null ? docText.slice(chunk.start, chunk.end!) : "");
  const rows: [string, string][] = [
    ["Chunk", `#${chunk.index}`],
    ["Type", chunk.type],
    ["Tokens", chunk.tokens.toLocaleString()],
    ["Characters", chunk.chars.toLocaleString()],
    ["PDF pages", chunk.pages.join(", ") || "n/a"],
    ["Printed pages", chunk.printed_pages.filter(Boolean).join(", ") || "n/a"],
    ["Section", chunk.section ?? "n/a"],
    ["Overlap with previous", `${chunk.overlap_prev_chars.toLocaleString()} chars`],
    ["Position in text", chunk.start != null ? `${chunk.start.toLocaleString()} – ${chunk.end!.toLocaleString()}` : "not located"],
    ["Ends mid-sentence", chunk.type === "table" ? "n/a (table)" : chunk.ends_mid_sentence ? "yes" : "no"],
  ];
  return (
    <div className="space-y-3">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-3">
        {rows.map(([k, v]) => (<div key={k}><dt className="text-dim">{k}</dt><dd className="font-mono">{v}</dd></div>))}
      </dl>
      <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface-2 p-3 font-mono text-xs">{body}</pre>
    </div>
  );
}
