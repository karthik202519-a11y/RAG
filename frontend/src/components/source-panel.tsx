"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, apiBlobUrl, type SearchHit } from "@/lib/api";
import { Button, Card, ErrorBox, Spinner } from "./ui";

const CONTEXT = 700; // characters shown before and after the matched chunk

/** The original document text around a retrieved chunk, with the chunk highlighted. */
export function SourcePanel({ hit, onClose }: { hit: SearchHit; onClose: () => void }) {
  const q = useQuery({
    queryKey: ["source", hit.document_id, hit.version_id],
    queryFn: () => api<{ text: string; filename: string; needs_vision_pages: number[] }>(
      `/documents/${hit.document_id}/content?version_id=${hit.version_id}`),
  });
  const [img, setImg] = useState<{ page: number; url: string } | null>(null);
  const [imgErr, setImgErr] = useState<unknown>(null);
  const located = hit.start != null && hit.end != null;
  const text = q.data?.text ?? "";
  const visionPages = hit.pages.filter((p) => q.data?.needs_vision_pages.includes(p));

  async function showPage(page: number) {
    setImgErr(null);
    try {
      setImg({ page, url: await apiBlobUrl(`/documents/${hit.document_id}/pages/${page}/image?version_id=${hit.version_id}`) });
    } catch (e) {
      setImgErr(e);
    }
  }

  return (
    <Card title={`Source · ${hit.citation}`} actions={<Button size="sm" variant="ghost" onClick={onClose}>Close</Button>}>
      {q.isLoading && <Spinner label="Loading the document" />}
      <ErrorBox error={q.error} />
      {q.data && (
        located ? (
          <div>
            <p className="mb-2 text-xs text-dim">Characters {hit.start!.toLocaleString()}–{hit.end!.toLocaleString()} of the extracted text. Highlighted text is the retrieved chunk.</p>
            <div className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface-2 p-3 font-mono text-xs leading-5" data-testid="source-text">
              {hit.start! > CONTEXT && "… "}
              {text.slice(Math.max(0, hit.start! - CONTEXT), hit.start!)}
              <mark className="rounded bg-accent-soft px-0.5 text-text outline outline-1 outline-accent">{text.slice(hit.start!, hit.end!)}</mark>
              {text.slice(hit.end!, hit.end! + CONTEXT)}
              {hit.end! + CONTEXT < text.length && " …"}
            </div>
          </div>
        ) : (
          <div>
            <p className="mb-2 text-xs text-warn">This chunk could not be located in the extracted text (its text was modified, for example a table with its caption added). Showing the chunk itself.</p>
            <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface-2 p-3 font-mono text-xs">{hit.text}</pre>
          </div>
        )
      )}
      {visionPages.length > 0 && (
        <div className="mt-3">
          <p className="mb-1 text-xs text-dim">This chunk is on a page with diagrams or charts. Their content is not in the text.</p>
          <div className="flex gap-2">{visionPages.map((p) => <Button key={p} size="sm" onClick={() => showPage(p)}>Show page {p} image</Button>)}</div>
          <ErrorBox error={imgErr} />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          {img && <img src={img.url} alt={`Rendered page ${img.page} of ${hit.filename}`} className="mt-2 max-h-[32rem] rounded border border-border" />}
        </div>
      )}
    </Card>
  );
}
