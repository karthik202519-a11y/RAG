"use client";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { BookOpen } from "lucide-react";
import { api, type Catalog } from "@/lib/api";
import { LESSONS } from "@/lib/learning";

export function LearningGuide() {
  const path = usePathname();
  const lesson = LESSONS[path];
  if (!lesson) return null;
  return <details key={path} className="learning-guide mb-4 rounded-xl border border-border bg-surface px-4 py-3">
    <summary className="cursor-pointer text-sm font-semibold"><BookOpen className="mr-2 inline size-4 text-accent" aria-hidden />{lesson.title}<span className="ml-3 text-xs font-normal text-dim">Steps & explanation</span></summary>
    <p className="mt-3 max-w-3xl text-sm text-dim">{lesson.why}</p>
    <ol className="my-4 grid gap-4 md:grid-cols-3">{lesson.steps.map((step,i)=><li key={step} className="text-sm"><span className="mb-1 block font-semibold text-accent">Step {i+1}</span>{step}</li>)}</ol>
    <p className="max-w-4xl text-sm text-warn">What to check: {lesson.check}</p>
  </details>;
}

export function MethodGuide({id}:{id:string}) {
  const catalog=useQuery({queryKey:["catalog"],queryFn:()=>api<Catalog>("/method-catalog"),staleTime:300000});
  const m=catalog.data?.items.find(m=>m.id===id);
  if(!m) return null;
  return <details className="rounded-lg bg-surface-2 p-3 text-sm" key={id}>
    <summary className="cursor-pointer font-semibold">How {m.name} works</summary>
    <p className="mt-2 leading-relaxed text-dim">{m.how_it_works}</p>
    <dl className="mt-3 space-y-2 text-xs"><div><dt className="font-semibold">Good for</dt><dd>{m.advantages.join("; ")}</dd></div><div><dt className="font-semibold">Tradeoffs</dt><dd>{m.disadvantages.join("; ")}</dd></div>{m.requires?.length>0&&<div><dt className="font-semibold">Prerequisites</dt><dd>{m.requires.join(", ")}</dd></div>}</dl>
  </details>;
}
