"use client";

import clsx from "clsx";
import { LearningGuide } from "./learning-guide";
import { AlertTriangle, Loader2 } from "lucide-react";
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";

export function Button({
  variant = "secondary", size = "md", loading, className, children, disabled, ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger"; size?: "sm" | "md"; loading?: boolean;
}) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-md font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60",
        size === "sm" ? "h-8 px-2.5 text-xs" : "h-9 px-3.5 text-sm",
        variant === "primary" && "bg-accent text-accent-ink hover:brightness-110 disabled:border disabled:border-border disabled:bg-surface-2 disabled:text-dim disabled:hover:brightness-100",
        variant === "secondary" && "border border-border bg-surface hover:bg-surface-2",
        variant === "ghost" && "hover:bg-surface-2",
        variant === "danger" && "border border-bad/40 bg-bad-soft text-bad hover:brightness-95",
        className,
      )}
    >
      {loading && <Loader2 className="size-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

export function Card({ title, actions, children, className, pad = true }: {
  title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; pad?: boolean;
}) {
  return (
    <section className={clsx("min-w-0 rounded-xl border border-border bg-surface", className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-2.5">
          <h2 className="text-sm font-semibold">{title}</h2>
          <div className="flex items-center gap-2">{actions}</div>
        </header>
      )}
      <div className={pad ? "p-4" : ""}>{children}</div>
    </section>
  );
}

const TONES = {
  neutral: "bg-surface-2 text-dim",
  good: "bg-good-soft text-good",
  warn: "bg-warn-soft text-warn",
  bad: "bg-bad-soft text-bad",
  accent: "bg-accent-soft text-accent",
} as const;

export function Badge({ tone = "neutral", children, title }: { tone?: keyof typeof TONES; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={clsx("inline-flex shrink-0 items-center whitespace-nowrap rounded px-1.5 py-0.5 text-xs font-medium", TONES[tone])}>
      {children}
    </span>
  );
}

export function statusTone(s: string): keyof typeof TONES {
  if (["ready", "succeeded", "implemented"].includes(s)) return "good";
  if (["failed", "dead"].includes(s)) return "bad";
  if (["processing", "running", "queued", "uploaded", "requires_credentials"].includes(s)) return "warn";
  return "neutral";
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block min-w-0 space-y-1">
      <span className="text-xs font-medium text-dim">{label}</span>
      {children}
      {hint && <span className="block text-xs leading-snug text-dim">{hint}</span>}
    </label>
  );
}

const control = "h-9 min-w-0 max-w-full w-full rounded-md border border-border bg-surface px-2.5 text-sm placeholder:text-dim";
export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={clsx(control, props.className)} />;
}
export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={clsx(control, props.className)} />;
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-surface px-4 py-3">
      <div className="text-xs text-dim">{label}</div>
      <div className="mt-0.5 font-mono text-2xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-dim">{sub}</div>}
    </div>
  );
}

/** Skeleton placeholder with an accessible status message: layout holds its shape while data arrives. */
export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="space-y-3 p-4">
      <p className="flex items-center gap-2 text-xs text-dim"><Loader2 className="size-3.5 animate-spin" aria-hidden /> {label}…</p>
      <div className="h-5 w-2/5 animate-pulse rounded bg-surface-2" />
      <div className="h-24 animate-pulse rounded-lg bg-surface-2" />
      <div className="h-4 w-3/5 animate-pulse rounded bg-surface-2" />
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className="flex gap-2 rounded-md border border-bad/40 bg-bad-soft px-3 py-2 text-sm text-bad">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
      <span className="break-words">{msg}</span>
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-border p-8 text-center">
      <p className="font-medium">{title}</p>
      {children && <div className="mt-1 text-sm text-dim">{children}</div>}
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <>
    <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 max-w-5xl text-sm leading-relaxed text-dim">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
    <LearningGuide />
    </>
  );
}

export const fmt = {
  n: (v: number | null | undefined) => (v == null ? "n/a" : v.toLocaleString()),
  bytes: (b: number) => (b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`),
  pct: (v: number | null | undefined) => (v == null ? "n/a" : `${(v * 100).toFixed(0)}%`),
};
