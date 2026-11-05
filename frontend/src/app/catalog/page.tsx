"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import { MethodDetail, StatusBadge } from "@/components/method-detail";
import { Card, Empty, ErrorBox, Input, PageHeader, Select, Spinner } from "@/components/ui";
import { api, type Catalog } from "@/lib/api";

export default function CatalogPage() {
  const [q, setQ] = useState("");
  const [category, setCategory] = useState("");
  const [status, setStatus] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const cat = useQuery({ queryKey: ["catalog"], staleTime: Infinity, queryFn: () => api<Catalog>("/method-catalog") });

  const items = useMemo(() => (cat.data?.items ?? []).filter((m) =>
    (!category || m.category === category) && (!status || m.status === status) &&
    (!q || `${m.name} ${m.summary} ${m.id}`.toLowerCase().includes(q.toLowerCase()))), [cat.data, q, category, status]);
  const names = useMemo(() => Object.fromEntries((cat.data?.items ?? []).map((m) => [m.id, m.name])), [cat.data]);
  const current = items.find((m) => m.id === selected) ?? items[0];

  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader title="Method Catalog" subtitle="Every method the studio knows about, generated from the backend registry. 'Runs for real' means it executes in this backend today; 'Not built yet' methods are described for learning but cannot be run." />
      <ErrorBox error={cat.error} />
      {cat.isLoading ? <Spinner /> : (
        <div className="grid gap-5 lg:grid-cols-[22rem_1fr]">
          <Card pad={false} className="h-fit">
            <div className="space-y-2 border-b border-border p-3">
              <div className="relative">
                <Search className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-dim" aria-hidden />
                <Input aria-label="Search methods" placeholder="Search methods" className="pl-8" value={q} onChange={(e) => setQ(e.target.value)} />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Select aria-label="Category" value={category} onChange={(e) => setCategory(e.target.value)}>
                  <option value="">All categories</option>
                  {Object.entries(cat.data?.categories ?? {}).map(([c, n]) => <option key={c} value={c}>{c} ({n})</option>)}
                </Select>
                <Select aria-label="Status" value={status} onChange={(e) => setStatus(e.target.value)}>
                  <option value="">Any status</option>
                  <option value="implemented">Runs for real</option>
                  <option value="unavailable">Not built yet</option>
                </Select>
              </div>
              <p className="text-xs text-dim">
                {cat.data?.counts.implemented} run for real · {cat.data?.counts.unavailable} not built yet
              </p>
            </div>
            <ul className="max-h-[70vh] divide-y divide-border overflow-y-auto">
              {items.map((m) => (
                <li key={m.id}>
                  <button onClick={() => setSelected(m.id)} aria-current={current?.id === m.id}
                    className={clsx("flex w-full items-start justify-between gap-2 px-3 py-2 text-left hover:bg-surface-2", current?.id === m.id && "bg-accent-soft")}>
                    <span>
                      <span className="block text-sm font-medium">{m.name}</span>
                      <span className="text-xs text-dim">{m.category}</span>
                    </span>
                    <StatusBadge m={m} />
                  </button>
                </li>
              ))}
              {items.length === 0 && <li className="p-4 text-sm text-dim">No methods match.</li>}
            </ul>
          </Card>
          <Card>{current ? <MethodDetail m={current} names={names} /> : <Empty title="Select a method" />}</Card>
        </div>
      )}
    </div>
  );
}
