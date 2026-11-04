"use client";

import { Check, Info, TriangleAlert, X } from "lucide-react";
import type { Method } from "@/lib/api";
import { Badge, statusTone } from "./ui";

export function StatusBadge({ m }: { m: Method }) {
  const label = m.status === "implemented" ? "Runs for real" : m.status === "requires_credentials" ? "Needs credentials" : "Not built yet";
  return <Badge tone={statusTone(m.status)} title={m.unavailable_reason ?? undefined}>{label}</Badge>;
}

/** The educational panel for one method. Content comes straight from the backend registry. */
export function MethodDetail({ m, names }: { m: Method; names?: Record<string, string> }) {
  return (
    <div className="space-y-4 text-sm">
      <div>
        <div className="mb-1 flex flex-wrap items-center gap-2">
          <h3 className="text-base font-semibold">{m.name}</h3>
          <StatusBadge m={m} />
        </div>
        <p>{m.summary}</p>
      </div>

      {m.status === "unavailable" && (
        <div className="flex gap-2 rounded-md border border-warn/40 bg-warn-soft px-3 py-2 text-warn" role="note">
          <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
          <span><strong>Cannot be run yet.</strong> {m.unavailable_reason}</span>
        </div>
      )}

      <section>
        <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">How does it work?</h4>
        <p className="text-dim">{m.how_it_works}</p>
      </section>

      <div className="grid gap-3 sm:grid-cols-2">
        <section>
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">Advantages</h4>
          <ul className="space-y-1">{m.advantages.map((a) => <li key={a} className="flex gap-1.5"><Check className="mt-0.5 size-3.5 shrink-0 text-good" aria-hidden />{a}</li>)}</ul>
        </section>
        <section>
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">Disadvantages</h4>
          <ul className="space-y-1">{m.disadvantages.map((a) => <li key={a} className="flex gap-1.5"><X className="mt-0.5 size-3.5 shrink-0 text-bad" aria-hidden />{a}</li>)}</ul>
        </section>
      </div>

      {m.params.length > 0 && (
        <section>
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">Parameters explained</h4>
          <dl className="space-y-1.5">
            {m.params.map((p) => (
              <div key={p.name}>
                <dt className="font-medium">{p.label} <span className="font-mono text-[11px] text-dim">{p.name}</span></dt>
                <dd className="text-dim">{p.help}{p.min != null && p.max != null && ` Range ${p.min}–${p.max}.`} Default {String(p.default)}.</dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      {(m.suitable_documents.length > 0 || m.use_cases.length > 0) && (
        <section>
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">When should I use it?</h4>
          <p className="text-dim">{[...m.suitable_documents, ...m.use_cases].join(" · ")}</p>
        </section>
      )}

      {(m.performance_notes || m.cost_notes) && (
        <section className="text-dim">
          {m.performance_notes && <p><span className="font-medium text-text">Performance:</span> {m.performance_notes}</p>}
          {m.cost_notes && <p><span className="font-medium text-text">Cost:</span> {m.cost_notes}</p>}
        </section>
      )}

      {m.limitations.length > 0 && (
        <section>
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">Known limitations</h4>
          <ul className="list-disc space-y-0.5 pl-5 text-dim">{m.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
        </section>
      )}

      {m.compare_with.length > 0 && (
        <p className="text-dim"><span className="font-medium text-text">Compare with:</span> {m.compare_with.map((id) => names?.[id] ?? id).join(", ")}</p>
      )}

      {m.recommended_experiment && (
        <div className="flex gap-2 rounded-md bg-accent-soft px-3 py-2 text-accent">
          <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
          <span><strong>Try this:</strong> {m.recommended_experiment}</span>
        </div>
      )}
    </div>
  );
}
