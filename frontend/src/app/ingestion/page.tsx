"use client";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, Trash2, UploadCloud, X } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Badge, Button, Card, Empty, ErrorBox, Field, Input, PageHeader, Select, Spinner, Stat, fmt, statusTone } from "@/components/ui";
import { API_URL, api, apiBlobUrl, getToken, type Collection, type Doc, type FiguresInfo, type IngestionStats, type Job } from "@/lib/api";
import { useProject } from "@/lib/project";

type UploadResult = { filename: string; status: string; error?: string; job_id?: string };

export default function IngestionPage() {
  const qc = useQueryClient();
  const { project, loading } = useProject();
  const pid = project?.id;
  const fileInput = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [staged, setStaged] = useState<File[]>([]);
  const [collectionId, setCollectionId] = useState("");
  const [tags, setTags] = useState("");
  const [parser, setParser] = useState("");
  const [results, setResults] = useState<UploadResult[]>([]);
  const [filters, setFilters] = useState({ q: "", status: "", ext: "" });
  const [openDoc, setOpenDoc] = useState<string | null>(null);
  const [liveJobs, setLiveJobs] = useState<Job[] | null>(null);

  const cols = useQuery({ queryKey: ["collections", pid], enabled: !!pid, queryFn: () => api<Collection[]>(`/projects/${pid}/collections`) });
  const stats = useQuery({ queryKey: ["stats", pid], enabled: !!pid, refetchInterval: 5000, queryFn: () => api<IngestionStats>(`/projects/${pid}/ingestion-stats`) });
  const fmts = useQuery({ queryKey: ["formats"], queryFn: () => api<{ extensions: string[]; max_upload_mb: number }>("/supported-formats") });
  const docs = useQuery({
    queryKey: ["docs", pid, filters], enabled: !!pid, refetchInterval: 4000, placeholderData: keepPreviousData,
    queryFn: () => api<{ total: number; items: Doc[] }>(`/projects/${pid}/documents?` + new URLSearchParams(Object.entries(filters).filter(([, v]) => v)).toString()),
  });
  const jobsQ = useQuery({ queryKey: ["jobs", pid], enabled: !!pid, queryFn: () => api<{ items: Job[] }>(`/projects/${pid}/ingestion-jobs?limit=100`) });

  // Live job updates over Server-Sent Events; falls back to polling when the stream cannot open.
  useEffect(() => {
    if (!pid) return;
    const token = getToken();
    if (!token) return;
    const es = new EventSource(`${API_URL}/projects/${pid}/ingestion-jobs-stream?token=${encodeURIComponent(token)}`);
    es.onmessage = (e) => {
      setLiveJobs(JSON.parse(e.data) as Job[]);
      void qc.invalidateQueries({ queryKey: ["docs", pid] });
      void qc.invalidateQueries({ queryKey: ["stats", pid] });
    };
    es.onerror = () => setLiveJobs(null);
    return () => es.close();
  }, [pid, qc]);

  const jobs = liveJobs ?? jobsQ.data?.items ?? [];

  const wantsParser = staged.some((f) => /[.](pdf|png|jpe?g)$/i.test(f.name));
  const upload = useMutation({
    mutationFn: async () => {
      const form = new FormData();
      staged.forEach((f) => form.append("files", f, f.name));
      if (collectionId) form.append("collection_id", collectionId);
      if (tags) form.append("tags", tags);
      if (parser && wantsParser) form.append("parser", parser);
      return api<{ results: UploadResult[] }>(`/projects/${pid}/documents`, { method: "POST", body: form });
    },
    onSuccess: (r) => {
      setResults(r.results); setStaged([]);
      void qc.invalidateQueries({ queryKey: ["docs", pid] });
      void qc.invalidateQueries({ queryKey: ["jobs", pid] });
    },
  });
  const jobAction = useMutation({
    mutationFn: ({ id, action }: { id: string; action: "cancel" | "retry" }) => api(`/ingestion-jobs/${id}/${action}`, { method: "POST" }),
    onSuccess: () => { void qc.invalidateQueries({ queryKey: ["jobs", pid] }); },
  });

  if (loading) return <Spinner />;
  if (!project) return <Empty title="No project selected"><Link className="text-accent underline" href="/projects">Create a project first</Link></Empty>;

  // Copy out of the live FileList immediately: resetting the input clears it before React runs the updater.
  const addFiles = (list: FileList | null) => {
    const picked = list ? Array.from(list) : [];
    if (picked.length) setStaged((s) => [...s, ...picked]);
  };
  const s = stats.data;
  const active = jobs.filter((j) => ["queued", "running"].includes(j.status));

  return (
    <div className="w-full min-w-0">
      <PageHeader title="Document Ingestion" subtitle="Upload files. Each one is hashed (duplicates are detected), stored, and parsed by a background job you can watch, cancel and retry." />

      <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Documents" value={fmt.n(s?.documents)} />
        <Stat label="Pages" value={fmt.n(s?.pages)} />
        <Stat label="Tokens" value={fmt.n(s?.tokens)} />
        <Stat label="Storage" value={s ? fmt.bytes(s.storage_bytes) : "n/a"} />
        <Stat label="Throughput" value={s?.throughput_pages_per_second != null ? `${s.throughput_pages_per_second} pages/s` : "n/a"}
          sub={s?.extraction_success_rate != null ? `${fmt.pct(s.extraction_success_rate)} success` : undefined} />
      </div>

      <div className="grid gap-5 lg:grid-cols-[21rem_minmax(0,1fr)]">
        <div className="min-w-0 space-y-5">
          <Card title="Upload">
            <div
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); addFiles(e.dataTransfer.files); }}
              className={`rounded-lg border-2 border-dashed p-6 text-center transition-colors ${drag ? "border-accent bg-accent-soft" : "border-border"}`}>
              <UploadCloud className="mx-auto mb-2 size-7 text-dim" aria-hidden />
              <p className="text-sm">Drag files here or</p>
              <Button className="mt-2" onClick={() => fileInput.current?.click()}>Choose files</Button>
              <input ref={fileInput} type="file" multiple hidden aria-label="Choose files to upload"
                accept={(fmts.data?.extensions ?? []).map((e) => `.${e}`).join(",")} onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
              <p className="mt-2 text-xs text-dim">{(fmts.data?.extensions ?? []).join(", ")} · up to {fmts.data?.max_upload_mb ?? "…"} MB each</p>
            </div>
            {staged.length > 0 && (
              <ul className="mt-3 space-y-1 text-sm">
                {staged.map((f, i) => (
                  <li key={`${f.name}-${i}`} className="flex items-center justify-between gap-2 rounded bg-surface-2 px-2 py-1">
                    <span className="truncate">{f.name}</span>
                    <span className="flex items-center gap-2 text-xs text-dim">{fmt.bytes(f.size)}
                      <button aria-label={`Remove ${f.name}`} onClick={() => setStaged((s) => s.filter((_, j) => j !== i))}><X className="size-3.5" /></button>
                    </span>
                  </li>
                ))}
              </ul>
            )}
            <div className="mt-3 grid grid-cols-2 gap-2">
              <Field label="Collection">
                <Select value={collectionId} onChange={(e) => setCollectionId(e.target.value)}>
                  <option value="">None</option>
                  {(cols.data ?? []).map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                </Select>
              </Field>
              <Field label="Tags (comma separated)"><Input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="spec, draft" /></Field>
              {wantsParser && (
                <Field label="PDF / image parser" hint={parser === "rapidocr" ? "Local OCR for scanned pages: slow, no text layer needed." : parser === "pdfplumber" ? "Finds ruled tables PyMuPDF can miss." : parser === "azure_document_intelligence" ? "Needs Document Intelligence credentials." : "PyMuPDF for PDFs; OCR for images."}>
                  <Select value={parser} onChange={(e) => setParser(e.target.value)} aria-label="Parser">
                    <option value="">Automatic</option><option value="pdfplumber">pdfplumber (tables)</option><option value="rapidocr">RapidOCR (scanned pages, images)</option>
                    <option value="azure_document_intelligence">Azure Document Intelligence</option>
                  </Select>
                </Field>)}
            </div>
            <Button variant="primary" className="mt-3 w-full" disabled={staged.length === 0} loading={upload.isPending} onClick={() => upload.mutate()}>
              Upload {staged.length > 0 ? `${staged.length} file${staged.length > 1 ? "s" : ""}` : ""}
            </Button>
            <div className="mt-2"><ErrorBox error={upload.error} /></div>
            {results.length > 0 && (
              <ul className="mt-3 space-y-1 text-xs" aria-label="Upload results">
                {results.map((r, i) => (
                  <li key={i} className="flex items-start justify-between gap-2">
                    <span className="truncate">{r.filename}</span>
                    <span className="text-right">
                      <Badge tone={r.status === "queued" ? "good" : r.status === "duplicate" ? "neutral" : "bad"}>{r.status}</Badge>
                      {r.error && <span className="block text-bad">{r.error}</span>}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title={`Jobs ${active.length ? `(${active.length} active)` : ""}`} pad={false}>
            {jobs.length === 0 ? <p className="p-4 text-sm text-dim">No jobs yet.</p> : (
              <ul className="max-h-96 divide-y divide-border overflow-y-auto">
                {jobs.slice(0, 50).map((j) => (
                  <li key={j.id} className="px-4 py-2.5 text-sm">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate">{j.filename ?? j.kind}</span>
                      <Badge tone={statusTone(j.status)}>{j.status}</Badge>
                    </div>
                    {(j.status === "running" || j.status === "queued") && (
                      <div className="mt-1.5">
                        <div className="h-1.5 overflow-hidden rounded bg-surface-2" role="progressbar" aria-valuenow={Math.round(j.progress * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={`Progress for ${j.filename}`}>
                          <div className="h-full bg-accent transition-all" style={{ width: `${Math.max(3, j.progress * 100)}%` }} />
                        </div>
                        <div className="mt-0.5 flex justify-between text-xs text-dim"><span>{j.stage ?? "waiting"}</span><span>attempt {j.attempts}/{j.max_attempts}</span></div>
                      </div>
                    )}
                    {j.error && <p className="mt-1 break-words text-xs text-bad">{j.error}</p>}
                    <div className="mt-1 flex gap-2">
                      {["queued", "running"].includes(j.status) && <Button size="sm" variant="ghost" onClick={() => jobAction.mutate({ id: j.id, action: "cancel" })}>Cancel</Button>}
                      {["dead", "cancelled", "failed"].includes(j.status) && <Button size="sm" onClick={() => jobAction.mutate({ id: j.id, action: "retry" })}>Retry</Button>}
                    </div>
                  </li>
                ))}
              </ul>
            )}
            <div className="px-4 pb-3"><ErrorBox error={jobAction.error} /></div>
          </Card>
        </div>

        <div className="min-w-0 space-y-5">
          <Card title={`Documents (${docs.data?.total ?? 0})`} pad={false}
            actions={
              <div className="flex min-w-0 flex-wrap gap-2">
                <Input aria-label="Search documents" placeholder="Search name" className="h-8 w-40" value={filters.q} onChange={(e) => setFilters({ ...filters, q: e.target.value })} />
                <Select aria-label="Filter by status" className="h-8 w-32" value={filters.status} onChange={(e) => setFilters({ ...filters, status: e.target.value })}>
                  <option value="">Any status</option>
                  {["ready", "processing", "uploaded", "failed"].map((x) => <option key={x}>{x}</option>)}
                </Select>
                <Select aria-label="Filter by type" className="h-8 w-28" value={filters.ext} onChange={(e) => setFilters({ ...filters, ext: e.target.value })}>
                  <option value="">Any type</option>
                  {Object.keys(s?.by_type ?? {}).map((x) => <option key={x}>{x}</option>)}
                </Select>
              </div>
            }>
            {docs.isLoading ? <Spinner /> : docs.data?.items.length === 0 ? (
              <div className="p-4"><Empty title="No documents match">Upload files on the left.</Empty></div>
            ) : (
              <div className="relative overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-border text-xs text-dim">
                    <tr><th className="px-4 py-2 font-medium">Name</th><th className="px-2 font-medium">Status</th><th className="px-2 text-right font-medium">Pages</th><th className="px-2 text-right font-medium">Tokens</th><th className="px-2 text-right font-medium">Tables</th><th className="px-2 font-medium">Ver.</th><th><span className="sr-only">Actions</span></th></tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {docs.data?.items.map((d) => {
                      const v = d.latest_version;
                      return (
                        <tr key={d.id} className="hover:bg-surface-2">
                          <td className="px-4 py-2">
                            <button className="flex items-center gap-2 text-left font-medium hover:underline" onClick={() => setOpenDoc(d.id)}>
                              <FileText className="size-4 shrink-0 text-dim" aria-hidden /><span className="truncate">{d.filename}</span>
                            </button>
                            {d.tags.length > 0 && <div className="mt-0.5 flex gap-1 pl-6">{d.tags.map((t) => <Badge key={t}>{t}</Badge>)}</div>}
                          </td>
                          <td className="px-2">{v && <Badge tone={statusTone(v.status)} title={v.error ?? undefined}>{v.status}</Badge>}</td>
                          <td className="px-2 text-right font-mono tabular-nums">{fmt.n(v?.pages)}</td>
                          <td className="px-2 text-right font-mono tabular-nums">{fmt.n(v?.token_count)}</td>
                          <td className="px-2 text-right font-mono tabular-nums">{fmt.n(v?.table_count)}</td>
                          <td className="px-2 font-mono text-xs text-dim">v{v?.version}</td>
                          <td className="px-2"><Link className="text-xs text-accent underline" href={`/chunking?doc=${d.id}`}>Chunk</Link></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
          {openDoc && <DocPanel id={openDoc} onClose={() => setOpenDoc(null)} />}
        </div>
      </div>
    </div>
  );
}

/** Diagram and chart pages: describe them with a vision model so their content becomes searchable. */
function Figures({ doc }: { doc: Doc }) {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["figures", doc.id], queryFn: () => api<FiguresInfo>(`/documents/${doc.id}/figures`), refetchInterval: (x) => (x.state.data?.pending ? 3000 : false) });
  const [open, setOpen] = useState<number | null>(null);
  const [img, setImg] = useState<{ page: number; url: string } | null>(null);
  const run = useMutation({
    mutationFn: (pages: number[] | null) => api<{ job_id: string }>(`/documents/${doc.id}/describe-figures`, { method: "POST", json: { pages } }),
    onSuccess: () => { void qc.invalidateQueries({ queryKey: ["figures", doc.id] }); void qc.invalidateQueries({ queryKey: ["jobs"] }); },
  });
  const f = q.data;
  if (!f || f.figures.length === 0) return null;
  const undescribed = f.figures.filter((x) => !x.described).map((x) => x.page);
  const cur = f.figures.find((x) => x.page === open);
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-dim">Figures ({f.figures.length} flagged · {f.described} described)</h3>
      <p className="text-xs text-dim">Pages with many drawn shapes. Their labels and arrows are not in the text until a vision model describes them.</p>
      <div className="flex flex-wrap gap-1">
        {f.figures.map((x) => (
          <button key={x.page} onClick={() => { setOpen(x.page); setImg(null); }} aria-label={`Figure page ${x.page}${x.described ? " described" : ""}`}
            className={`rounded border px-1.5 py-0.5 font-mono text-xs ${open === x.page ? "border-accent bg-accent-soft" : "border-border"} ${x.described ? "text-good" : "text-dim"}`}>
            {x.page}{x.described ? " ✓" : ""}
          </button>
        ))}
      </div>
      {cur && (
        <div className="rounded-md border border-border p-2 text-xs">
          {cur.described ? <p className="max-h-40 overflow-auto whitespace-pre-wrap">{cur.description}</p> : <p className="text-dim">Not described yet.</p>}
          <div className="mt-2 flex gap-2">
            <Button size="sm" onClick={async () => setImg({ page: cur.page, url: await apiBlobUrl(`/documents/${doc.id}/pages/${cur.page}/image`) })}>Show page image</Button>
            <Button size="sm" variant="primary" loading={run.isPending} disabled={!f.vision.configured} onClick={() => run.mutate([cur.page])}>{cur.described ? "Describe again" : "Describe this page"}</Button>
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          {img?.page === cur.page && <img src={img.url} alt={`Rendered page ${cur.page}`} className="mt-2 max-h-72 rounded border border-border" />}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="primary" disabled={!f.vision.configured || undescribed.length === 0} loading={run.isPending}
          onClick={() => { if (window.confirm(`Describe ${undescribed.length} page(s) with the vision model? This uses roughly 1,000 input tokens per page.`)) run.mutate(null); }}>
          Describe {undescribed.length} remaining page{undescribed.length === 1 ? "" : "s"}
        </Button>
        {!f.vision.configured && <span className="text-xs text-warn">Vision model not configured: {f.vision.detail}</span>}
      </div>
      <ErrorBox error={run.error} />
    </div>
  );
}

/** Tags and custom metadata. State starts from the document; remounted per document via `key`. */
function MetaEditor({ doc }: { doc: Doc }) {
  const qc = useQueryClient();
  const [tags, setTags] = useState(doc.tags.join(", "));
  const [meta, setMeta] = useState(JSON.stringify(doc.doc_metadata, null, 2));
  const save = useMutation({
    mutationFn: () => {
      let parsed: Record<string, unknown>;
      try { parsed = JSON.parse(meta || "{}"); } catch { throw new Error("Metadata is not valid JSON"); }
      return api(`/documents/${doc.id}`, { method: "PATCH", json: { tags: tags.split(",").map((t) => t.trim()).filter(Boolean), doc_metadata: parsed } });
    },
    onSuccess: () => { void qc.invalidateQueries({ queryKey: ["doc", doc.id] }); void qc.invalidateQueries({ queryKey: ["docs"] }); },
  });
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-dim">Metadata</h3>
      <Field label="Tags (comma separated)"><Input value={tags} onChange={(e) => setTags(e.target.value)} /></Field>
      <Field label="Custom metadata (JSON)">
        <textarea value={meta} onChange={(e) => setMeta(e.target.value)} rows={4} spellCheck={false}
          className="w-full rounded-md border border-border bg-surface p-2 font-mono text-xs" />
      </Field>
      <ErrorBox error={save.error} />
      <Button variant="primary" size="sm" loading={save.isPending} onClick={() => save.mutate()}>Save metadata</Button>
    </div>
  );
}

function DocPanel({ id, onClose }: { id: string; onClose: () => void }) {
  const qc = useQueryClient();
  const d = useQuery({ queryKey: ["doc", id], queryFn: () => api<Doc>(`/documents/${id}`), refetchInterval: 4000 });
  const ready = d.data?.latest_version?.status === "ready";
  const content = useQuery({ queryKey: ["doc-content", id, d.data?.latest_version?.id], enabled: ready, queryFn: () => api<{ text: string; pages: number; needs_vision_pages: number[] }>(`/documents/${id}/content`) });
  const del = useMutation({
    mutationFn: () => api(`/documents/${id}`, { method: "DELETE" }),
    onSuccess: () => { void qc.invalidateQueries({ queryKey: ["docs"] }); void qc.invalidateQueries({ queryKey: ["stats"] }); onClose(); },
  });

  if (d.isLoading) return <Card><Spinner /></Card>;
  if (!d.data) return <Card><ErrorBox error={d.error} /></Card>;
  const doc = d.data;
  return (
    <Card title={doc.filename} actions={<Button size="sm" variant="ghost" onClick={onClose} aria-label="Close details"><X className="size-4" /></Button>}>
      <div className="grid gap-5 lg:grid-cols-2">
        <div className="space-y-4">
          <div>
            <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">Versions</h3>
            <table className="w-full text-xs">
              <tbody className="divide-y divide-border">
                {doc.versions?.slice().reverse().map((v) => (
                  <tr key={v.id}>
                    <td className="py-1.5 font-mono">v{v.version}</td>
                    <td><Badge tone={statusTone(v.status)}>{v.status}</Badge></td>
                    <td className="text-dim">{v.parser ?? "-"}</td>
                    <td className="text-right text-dim">{fmt.bytes(v.size_bytes)}</td>
                    <td className="text-right font-mono text-dim" title={v.sha256}>{v.sha256.slice(0, 8)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {doc.latest_version?.error && <div className="mt-2"><ErrorBox error={new Error(doc.latest_version.error)} /></div>}
          </div>
          <MetaEditor key={doc.id} doc={doc} />
          {ready && <Figures key={`fig-${doc.id}`} doc={doc} />}
          <div>
            <Button variant="danger" size="sm" loading={del.isPending}
              onClick={() => { if (window.confirm(`Delete "${doc.filename}" and all its versions?`)) del.mutate(); }}><Trash2 className="size-3.5" /> Delete document</Button>
            <ErrorBox error={del.error} />
          </div>
        </div>
        <div>
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-dim">Extracted text preview</h3>
          {!ready ? <p className="text-sm text-dim">Available once extraction is ready.</p> : content.isLoading ? <Spinner /> : (
            <>
              <p className="mb-1 text-xs text-dim">{content.data?.pages} page(s){content.data?.needs_vision_pages.length ? ` · ${content.data.needs_vision_pages.length} diagram/chart page(s) flagged` : ""} · showing first 4,000 characters</p>
              <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface-2 p-3 font-mono text-xs">{content.data?.text.slice(0, 4000)}</pre>
            </>
          )}
        </div>
      </div>
    </Card>
  );
}
