"use client";
import Link from "next/link";
import { LESSONS } from "@/lib/learning";
import { PageHeader } from "@/components/ui";
export default function Learn(){return <div className="mx-auto max-w-5xl"><PageHeader title="Understand every stage" subtitle="Follow the data from a document to a grounded answer. Each lesson links to a working tool and explains what to inspect."/><div className="space-y-8">{Object.entries(LESSONS).map(([href,l])=><section key={href} className="border-t border-border pt-5"><h2 className="text-xl font-semibold"><Link href={href} className="text-accent hover:underline">{l.title}</Link></h2><p className="mt-2 text-sm text-dim">{l.why}</p><ol className="my-4 list-decimal space-y-2 pl-5 text-sm">{l.steps.map(s=><li key={s}>{s}</li>)}</ol><p className="text-sm text-warn">Check: {l.check}</p></section>)}</div></div>}
