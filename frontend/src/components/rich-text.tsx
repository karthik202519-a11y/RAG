"use client";

import { useMemo } from "react";

type Block = { kind: "text"; text: string } | { kind: "table"; head: string[]; rows: string[][] };

const cells = (line: string) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
const isSeparator = (line: string) => /^\|?[\s:|-]+\|?$/.test(line.trim()) && line.includes("-");

/** Split chunk text into prose and Markdown tables so tables can be drawn as tables. */
export function parseBlocks(text: string): Block[] {
  const lines = text.split("\n");
  const out: Block[] = [];
  let prose: string[] = [];
  const flush = () => {
    if (prose.join("").trim()) out.push({ kind: "text", text: prose.join("\n") });
    prose = [];
  };
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trimStart().startsWith("|") && i + 1 < lines.length && isSeparator(lines[i + 1])) {
      flush();
      const head = cells(lines[i]);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && lines[i].trimStart().startsWith("|")) rows.push(cells(lines[i++]));
      i--;
      out.push({ kind: "table", head, rows });
    } else {
      prose.push(lines[i]);
    }
  }
  flush();
  return out;
}

export function ChunkText({ text, maxRows = 5, className }: { text: string; maxRows?: number; className?: string }) {
  const blocks = useMemo(() => parseBlocks(text), [text]);
  return (
    <div className={className}>
      {blocks.map((b, i) =>
        b.kind === "text" ? (
          <p key={i} className="whitespace-pre-wrap text-sm">{b.text}</p>
        ) : (
          <div key={i} className="my-1 overflow-x-auto rounded border border-border">
            <table className="w-full text-left text-xs">
              <thead className="bg-surface-2 text-dim"><tr>{b.head.map((h, j) => <th key={j} className="px-2 py-1 font-medium">{h}</th>)}</tr></thead>
              <tbody className="divide-y divide-border">
                {b.rows.slice(0, maxRows).map((r, k) => <tr key={k}>{r.map((c, j) => <td key={j} className="px-2 py-1 align-top">{c}</td>)}</tr>)}
              </tbody>
            </table>
            {b.rows.length > maxRows && <p className="border-t border-border px-2 py-1 text-xs text-dim">+ {b.rows.length - maxRows} more row{b.rows.length - maxRows === 1 ? "" : "s"}</p>}
          </div>
        ),
      )}
    </div>
  );
}
