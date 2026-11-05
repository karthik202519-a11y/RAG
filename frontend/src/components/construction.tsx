"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Badge, ErrorBox, Spinner } from "./ui";

type Evidence={provider:string;model:string;store:string;dimension:number;chunk_count:number;tokens_billed:number;cache_hits:number;chunking:{method:string};samples:{id:string;filename:string;pages:number[];text:string;embedding_input:string;vector_preview:number[];dimension:number}[];graph:{status:string;model:string;chunks_processed:number;relations_dropped:number;grounding:string;edges:{source:string;relation:string;target:string;filename:string;chunk_id:string;text:string}[]}};
export function Construction({indexId,kind}:{indexId:string;kind:"graph"|"embeddings"}) {
  const q=useQuery({queryKey:["construction",indexId],enabled:!!indexId,queryFn:()=>api<Evidence>(`/indexes/${indexId}/construction`),refetchInterval:kind==="graph"?10000:false});
  const d=q.data;
  return <details className="mb-4 rounded-xl border border-border bg-surface p-4" aria-label="Construction evidence">
    <summary className="cursor-pointer text-sm font-semibold">{kind==="graph"?"From source text to graph edges":"From source text to a plotted point"}</summary>
    <p className="mt-1 text-sm text-dim">Actual artifacts from this index. Expand a sample to follow its construction.</p>
    <ErrorBox error={q.error}/>{q.isLoading&&<Spinner label="Reading construction evidence"/>}
    {d&&<>
      <div className="my-4 flex flex-wrap items-center gap-2 text-sm"><Badge>{d.chunking.method}</Badge><span aria-hidden>→</span><Badge>{d.provider} / {d.model}</Badge><span aria-hidden>→</span><Badge>{d.dimension} dimensions in {d.store}</Badge><span aria-hidden>→</span><Badge>{kind==="graph"?"Extract -> ground -> connect -> group":"Sample -> center -> PCA -> plot"}</Badge></div>
      {kind==="embeddings"?<><p className="mb-3 text-xs text-dim">{d.chunk_count} chunks · {d.cache_hits} cached embeddings · {d.tokens_billed} billed embedding tokens. PCA changes only the visualization; stored vectors are preserved.</p>{d.samples.map(s=><details key={s.id} className="border-t border-border py-3"><summary className="cursor-pointer text-sm font-medium">{s.filename} · pages {s.pages.join(", ")||"not paginated"} · {s.dimension} numbers</summary><div className="mt-3 grid gap-4 lg:grid-cols-2"><div><h3 className="text-xs font-semibold">Text sent to the embedding model</h3><pre className="mt-1 whitespace-pre-wrap break-words text-xs text-dim">{s.embedding_input}</pre></div><div><h3 className="text-xs font-semibold">First 12 stored coordinates</h3><p className="mt-1 break-words font-mono text-xs">[{s.vector_preview.join(", ")}, …]</p><p className="mt-2 text-xs text-dim">The full vector is used for similarity. The scatter plot below is computed from a sample of those full vectors.</p></div></div></details>)}</>:<>
        <p className="mb-3 text-xs text-dim">Graph: {d.graph.status} · {d.graph.chunks_processed} chunks processed · {d.graph.relations_dropped} proposals rejected · {d.graph.model||"no extraction model yet"}</p>
        <p className="mb-3 text-sm text-warn">{d.graph.grounding}</p>
        {!d.graph.edges.length&&<p className="text-sm text-dim">Build a graph below to see real relations and their source evidence here.</p>}
        {d.graph.edges.map((edge,i)=><details key={i} className="border-t border-border py-3"><summary className="cursor-pointer text-sm"><strong>{edge.source}</strong> · {edge.relation} · <strong>{edge.target}</strong><span className="ml-2 text-xs text-dim">{edge.filename}</span></summary><p className="mt-2 whitespace-pre-wrap text-sm text-dim">{edge.text}</p><p className="mt-2 font-mono text-xs text-dim">Source chunk: {edge.chunk_id}</p></details>)}
      </>}
    </>}
  </details>;
}
