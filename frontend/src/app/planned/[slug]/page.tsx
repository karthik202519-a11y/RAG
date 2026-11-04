"use client";

import { Wrench } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Badge, Card, Empty, PageHeader } from "@/components/ui";
import { PLANNED } from "@/lib/planned";

export default function PlannedPage() {
  const { slug } = useParams<{ slug: string }>();
  const p = PLANNED.find((x) => x.slug === slug);
  if (!p) return <Empty title="Unknown screen"><Link className="text-accent underline" href="/">Back to the dashboard</Link></Empty>;
  return (
    <div className="mx-auto max-w-3xl">
      <PageHeader title={p.title} subtitle={p.summary} actions={<Badge tone="warn">Not built yet · {p.stage}</Badge>} />
      <div className="space-y-5">
        <Card title="Why you cannot use this yet">
          <div className="flex gap-3 text-sm">
            <Wrench className="mt-0.5 size-4 shrink-0 text-warn" aria-hidden />
            <p>This screen is on the build plan, and there is no working feature behind it. It is listed here so the full scope stays visible, not to look finished.</p>
          </div>
        </Card>
        <Card title="What it will do">
          <ul className="list-disc space-y-1 pl-5 text-sm">{p.willDo.map((w) => <li key={w}>{w}</li>)}</ul>
        </Card>
        <Card title="What exists today"><p className="text-sm">{p.backendToday}</p></Card>
      </div>
    </div>
  );
}
