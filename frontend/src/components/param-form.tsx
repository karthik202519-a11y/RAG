"use client";

import type { Method, ParamSpec } from "@/lib/api";
import { Field, Input, Select } from "./ui";

/** The default value of every parameter of a method, straight from the registry. */
export function defaults(m: Method): Record<string, unknown> {
  return Object.fromEntries(m.params.map((p) => [p.name, p.default]));
}

/** Client-side mirror of the backend's rules, so the user sees the problem before running. */
export function validate(m: Method | undefined, v: Record<string, unknown>): string | null {
  if (!m) return "Pick a method";
  if (m.status === "unavailable") return m.unavailable_reason;
  for (const p of m.params) {
    const x = v[p.name];
    if ((p.type === "int" || p.type === "float") && (typeof x !== "number" || Number.isNaN(x))) return `${p.label}: enter a number`;
    if (typeof x === "number") {
      if (p.min != null && x < p.min) return `${p.label}: minimum is ${p.min}`;
      if (p.max != null && x > p.max) return `${p.label}: maximum is ${p.max}`;
    }
  }
  if (typeof v.overlap === "number" && typeof v.size === "number" && v.overlap >= v.size) return `Overlap (${v.overlap}) must be smaller than chunk size (${v.size}).`;
  if (typeof v.stride === "number" && typeof v.size === "number" && v.stride > v.size) return `Stride (${v.stride}) larger than the window (${v.size}) would skip text.`;
  return null;
}

export function ParamControl({ p, value, onChange }: { p: ParamSpec; value: unknown; onChange: (v: unknown) => void }) {
  if (p.type === "bool") {
    return (
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} /> {p.label}
        <span className="sr-only">{p.help}</span>
      </label>
    );
  }
  return (
    <Field label={`${p.label}`} hint={p.help}>
      {p.type === "select" ? (
        <Select value={String(value)} onChange={(e) => onChange(e.target.value)}>{p.options!.map((o) => <option key={o}>{o}</option>)}</Select>
      ) : (
        <Input type="number" value={value === "" || value == null ? "" : String(value)} min={p.min ?? undefined} max={p.max ?? undefined}
          step={p.type === "float" ? p.step ?? 0.05 : 1}
          onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))} />
      )}
    </Field>
  );
}
