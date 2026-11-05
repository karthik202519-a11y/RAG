"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

const STAGES = [
  { href: "/ingestion", label: "Documents", help: "Upload your sources" },
  { href: "/parsing", label: "Inspect", help: "Check extracted content" },
  { href: "/chunking", label: "Chunk", help: "Compare how text is split" },
  { href: "/indexes", label: "Index", help: "Choose embeddings and storage" },
  { href: "/retrieval", label: "Ask", help: "Search and inspect citations" },
  { href: "/evaluation", label: "Evaluate", help: "Measure answer quality" },
];

export function Workflow() {
  const path = usePathname();
  if (!STAGES.some((s) => s.href === path)) return null;
  return <nav aria-label="RAG learning workflow" className="mb-4 border-b border-border pb-3">
    <ol className="flex flex-wrap gap-1">
      {STAGES.map((stage, i) => <li key={stage.href}>
        <Link href={stage.href} title={stage.help} aria-current={path === stage.href ? "step" : undefined}
          className={clsx("flex min-h-9 items-center gap-2 rounded-md px-2.5 py-1.5 text-sm hover:bg-surface-2", path === stage.href ? "bg-accent-soft text-accent" : "text-dim")}>
          <span className="tabular-nums" aria-hidden>{i + 1}.</span>
          <span><span className="block font-medium">{stage.label}</span></span>
        </Link>
      </li>)}
    </ol>
  </nav>;
}
