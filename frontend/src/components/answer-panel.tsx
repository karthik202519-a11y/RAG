"use client";

import clsx from "clsx";
import { AlertTriangle, CheckCircle2, ShieldQuestion } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { Fragment, useState } from "react";
import { api, type AnswerResponse } from "@/lib/api";
import { Badge, Button, Card, ErrorBox } from "./ui";
import type { SourceRef } from "./source-panel";

/** Render the answer with each [n] marker as a button that opens that passage in its source. */
function AnswerText({ r, onOpen }: { r: AnswerResponse; onOpen: (s: SourceRef) => void }) {
  const parts = r.answer.split(/(\[\d{1,3}\])/g);
  return (
    <p className="whitespace-pre-wrap text-[15px] leading-7" data-testid="answer-text">
      {parts.map((p, i) => {
        const m = /^\[(\d{1,3})\]$/.exec(p);
        if (!m) return <Fragment key={i}>{p}</Fragment>;
        const n = Number(m[1]);
        const passage = r.context_passages.find((x) => x.n === n);
        return passage ? (
          <button key={i} onClick={() => onOpen(passage)} title={passage.citation}
            className="mx-0.5 rounded bg-accent-soft px-1 font-mono text-xs font-semibold text-accent hover:brightness-95">[{n}]</button>
        ) : (
          <span key={i} title="This number does not match any passage that was sent" className="mx-0.5 rounded bg-bad-soft px-1 font-mono text-xs font-semibold text-bad">[{n}?]</span>
        );
      })}
    </p>
  );
}

type Judged = { claims: { claim: string; verdict: "supported" | "contradicted" | "unsupported"; passage: number | null; quote: string | null }[];
  faithfulness: number | null; supported: number; claims_total: number; contradicted: number; relevance: string | null; judge_model: string; note: string };
const VERDICT_TONE = { supported: "good", contradicted: "bad", unsupported: "warn" } as const;

function Judge({ r, indexId }: { r: AnswerResponse; indexId: string }) {
  const judge = useMutation({ mutationFn: () => api<Judged>(`/indexes/${indexId}/judge`, { method: "POST", json: {
    question: r.question, answer: r.answer, passages: r.context_passages.map((p) => ({ n: p.n, text: p.text })) } }) });
  const j = judge.data;
  return (
    <div className="mt-4 border-t border-border pt-3 text-sm">
      <Button size="sm" onClick={() => judge.mutate()} disabled={judge.isPending}>{judge.isPending ? "Checking claims…" : j ? "Check again" : "Check claims against the passages"}</Button>
      <ErrorBox error={judge.error} />
      {j && (
        <div className="mt-3 space-y-2">
          <p>
            <span className="font-semibold">{j.supported} of {j.claims_total}</span> claims supported
            {j.faithfulness != null && <> ({Math.round(j.faithfulness * 100)}%)</>}
            {j.contradicted > 0 && <span className="text-bad"> · {j.contradicted} contradicted</span>}
            {j.relevance && <> · answers the question: {j.relevance}</>}
          </p>
          <ul className="space-y-1.5">
            {j.claims.map((c, i) => (
              <li key={i} className="flex gap-2">
                <Badge tone={VERDICT_TONE[c.verdict]}>{c.verdict}</Badge>
                <span>{c.claim}{c.quote && <span className="block text-xs text-dim">[{c.passage}] “{c.quote}”</span>}</span>
              </li>
            ))}
          </ul>
          <p className="text-xs text-dim">{j.note} Judge: {j.judge_model}.</p>
        </div>
      )}
    </div>
  );
}

export function AnswerPanel({ r, onOpen, indexId }: { r: AnswerResponse; onOpen: (s: SourceRef) => void; indexId?: string }) {
  const [showPrompt, setShowPrompt] = useState(false);
  const c = r.citations;
  return (
    <Card title={<span className="flex items-center gap-2">Answer <span className="font-normal text-dim">· {r.usage?.served_by ?? r.model}</span></span>}
      actions={r.abstained ? <Badge tone="warn">abstained</Badge> : c.has_citations && c.all_resolve ? <Badge tone="good">citations resolve</Badge> : undefined}>
      {r.abstained && (
        <div className="mb-3 flex gap-2 rounded-md bg-warn-soft px-3 py-2 text-sm text-warn" role="note">
          <ShieldQuestion className="mt-0.5 size-4 shrink-0" aria-hidden />
          <span>{r.abstain_reason ?? "The model said the passages do not contain the answer."}</span>
        </div>
      )}
      {r.empty && <p className="mb-2 text-sm text-warn">The model returned no text. Reasoning models spend output tokens thinking; raise the output limit.</p>}
      {r.truncated && <p className="mb-2 text-sm text-warn">The answer was cut off by the output-token limit.</p>}
      <AnswerText r={r} onOpen={onOpen} />

      {r.prompt && (
        <div className="mt-4 space-y-2 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            {c.has_citations ? (
              c.all_resolve ? <span className="flex items-center gap-1 text-good"><CheckCircle2 className="size-4" aria-hidden /> Every citation [{c.cited.join("], [")}] points at a passage that was sent.</span>
                : <span className="flex items-center gap-1 text-bad"><AlertTriangle className="size-4" aria-hidden /> Cites {c.unresolved.map((n) => `[${n}]`).join(", ")}, which was not sent.</span>
            ) : !r.abstained && <span className="flex items-center gap-1 text-warn"><AlertTriangle className="size-4" aria-hidden /> The answer has no citations.</span>}
          </div>
          <p className="text-xs text-dim">A resolving citation proves the number exists, not that the passage supports the claim. Read the source.</p>
        </div>
      )}

      {indexId && r.prompt && !r.abstained && r.answer && <Judge key={r.answer} r={r} indexId={indexId} />}

      {r.usage && (
        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-4">
          {[["Prompt tokens", r.usage.prompt_tokens], ["Answer tokens", r.usage.completion_tokens],
            ["Reasoning tokens", r.usage.reasoning_tokens], ["Finish", r.usage.finish_reason ?? "n/a"],
            ["Retrieval", `${r.timings_ms.retrieval_ms} ms`], ["Model", `${r.timings_ms.llm_ms} ms`],
            ["Cost", r.usage.estimated_cost_usd != null ? `$${r.usage.estimated_cost_usd}` : "enter prices to estimate"]].map(([k, v]) => (
            <div key={String(k)}><dt className="text-dim">{k}</dt><dd className="font-mono">{String(v)}</dd></div>
          ))}
        </dl>
      )}

      {r.prompt && (
        <div className="mt-4 border-t border-border pt-3">
          <Button size="sm" onClick={() => setShowPrompt(!showPrompt)} aria-expanded={showPrompt}>{showPrompt ? "Hide" : "Show"} the exact prompt</Button>
          {showPrompt && (
            <div className="mt-3 space-y-3">
              <p className="text-xs text-dim">System {r.prompt.system_tokens} tok · context {r.prompt.context_tokens} tok (budget {r.prompt.context_budget}) · question {r.prompt.question_tokens} tok
                · {r.context_passages.length} passage{r.context_passages.length === 1 ? "" : "s"} sent{r.dropped_passages.length > 0 ? `, ${r.dropped_passages.length} dropped for budget` : ""}</p>
              {r.prompt.messages.map((m, i) => (
                <div key={i}>
                  <div className={clsx("mb-1 text-xs font-semibold uppercase tracking-wide", m.role === "system" ? "text-warn" : "text-accent")}>{m.role}</div>
                  <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface-2 p-3 font-mono text-xs">{m.content}</pre>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
