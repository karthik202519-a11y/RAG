"use client";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { api } from "@/lib/api";
import { downloadRun, type WorkbenchRun } from "@/lib/workbench";
import { Badge, Button, ErrorBox, Spinner, statusTone } from "./ui";
export function ExecutionPanel({id}:{id:string}){
  const [downloadError,setDownloadError]=useState<unknown>(null);
  const q=useQuery({queryKey:["workbench-run",id],enabled:!!id,queryFn:()=>api<WorkbenchRun>(`/workbench-runs/${id}`),refetchInterval:q=>["queued","running"].includes(q.state.data?.status??"")?1200:false});
  if(q.isLoading)return <Spinner label="Loading execution"/>;
  const r=q.data;
  return <section className="space-y-4 rounded-xl border border-border bg-surface p-5"><ErrorBox error={q.error??downloadError}/>{r&&<>
    <div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-lg font-semibold">{r.name}</h2><p className="text-xs text-dim">Saved execution · {new Date(r.created_at).toLocaleString()}</p></div><div className="flex items-center gap-2"><Badge tone={statusTone(r.status==="done"?"succeeded":r.status)}>{r.status}</Badge><Button size="sm" onClick={()=>downloadRun(r.id).catch(setDownloadError)}>Export report</Button>{["queued","running"].includes(r.status)&&r.job&&<Button size="sm" onClick={()=>api(`/ingestion-jobs/${r.job!.id}/cancel`,{method:"POST"}).then(()=>q.refetch()).catch(setDownloadError)}>Cancel</Button>}</div></div>
    <ErrorBox error={r.error}/>{r.job&&["queued","running"].includes(r.status)&&<p role="status" className="text-sm text-accent">{r.job.stage||"Starting"} · {Math.round(r.job.progress*100)}%</p>}
    {r.artifacts.length===0?<p className="text-sm text-dim">Stage evidence will appear as execution starts.</p>:<ol className="divide-y divide-border">{r.artifacts.map((a,i)=><li key={i} className="py-3"><details><summary className="cursor-pointer text-sm"><span className="font-semibold">{a.stage}</span><span className="ml-3 text-dim">{a.status}{a.ms!=null?` · ${a.ms} ms`:""}</span></summary><pre className="mt-2 whitespace-pre-wrap break-words text-xs text-dim">{JSON.stringify(a.detail,null,2)}</pre></details></li>)}</ol>}
    {r.result?.cases.map(c=><div key={c.index_id} className="border-t border-border pt-4"><h3 className="font-semibold">{c.name} v{c.version}</h3><Link className="text-sm text-accent underline" href={`/retrieval?index=${c.index_id}`}>Inspect this index and its sources</Link>{c.answer&&<p className="my-3 whitespace-pre-wrap text-sm">{c.answer.answer}</p>}{c.retrieval?.results.map(h=><details key={h.chunk_id} className="py-2"><summary className="cursor-pointer text-sm">#{h.rank} · {h.citation} · score {h.score.toFixed(4)}</summary><p className="mt-2 whitespace-pre-wrap text-sm text-dim">{h.text}</p></details>)}</div>)}
  </>}</section>;
}
