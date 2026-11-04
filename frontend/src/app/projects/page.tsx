"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button, Card, Empty, ErrorBox, Field, Input, PageHeader, Spinner } from "@/components/ui";
import { api, type Collection, type Workspace } from "@/lib/api";
import { useProject } from "@/lib/project";

export default function ProjectsPage() {
  const qc = useQueryClient();
  const { projects, project, setProjectId, loading } = useProject();
  const ws = useQuery({ queryKey: ["workspaces"], queryFn: () => api<Workspace[]>("/workspaces") });
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [colName, setColName] = useState("");

  const create = useMutation({
    mutationFn: () => api<{ id: string }>("/projects", { method: "POST", json: { workspace_id: ws.data![0].id, name, description: desc } }),
    onSuccess: async (p) => {
      setName(""); setDesc("");
      await qc.invalidateQueries({ queryKey: ["projects"] });
      setProjectId(p.id);
    },
  });
  const cols = useQuery({
    queryKey: ["collections", project?.id], enabled: !!project,
    queryFn: () => api<Collection[]>(`/projects/${project!.id}/collections`),
  });
  const addCol = useMutation({
    mutationFn: () => api(`/projects/${project!.id}/collections`, { method: "POST", json: { name: colName } }),
    onSuccess: () => { setColName(""); void qc.invalidateQueries({ queryKey: ["collections", project?.id] }); },
  });
  const del = useMutation({
    mutationFn: (id: string) => api(`/projects/${id}`, { method: "DELETE" }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["projects"] }),
  });

  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title="Projects & Collections" subtitle="A project holds documents, chunking runs and (later) experiments. Collections group documents inside a project." />
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Your projects">
          {loading ? <Spinner /> : projects.length === 0 ? <Empty title="No projects yet">Create your first one on the right.</Empty> : (
            <ul className="divide-y divide-border">
              {projects.map((p) => (
                <li key={p.id} className="flex items-center justify-between gap-3 py-2.5">
                  <div className="min-w-0">
                    <div className="truncate font-medium">{p.name} {project?.id === p.id && <span className="ml-1 text-xs text-accent">active</span>}</div>
                    <div className="truncate text-xs text-dim">{p.description || "No description"}</div>
                  </div>
                  <div className="flex gap-2">
                    {project?.id !== p.id && <Button size="sm" onClick={() => setProjectId(p.id)}>Open</Button>}
                    <Button size="sm" variant="danger" loading={del.isPending}
                      onClick={() => { if (window.confirm(`Delete "${p.name}" and all its documents? This cannot be undone.`)) del.mutate(p.id); }}>
                      Delete
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
          <ErrorBox error={del.error} />
        </Card>

        <div className="space-y-5">
          <Card title="New project">
            <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
              <Field label="Name"><Input value={name} onChange={(e) => setName(e.target.value)} required maxLength={200} /></Field>
              <Field label="Description (optional)"><Input value={desc} onChange={(e) => setDesc(e.target.value)} /></Field>
              <ErrorBox error={create.error} />
              <Button type="submit" variant="primary" loading={create.isPending} disabled={!ws.data}>Create project</Button>
            </form>
          </Card>

          {project && (
            <Card title={`Collections in "${project.name}"`}>
              <ul className="mb-3 space-y-1 text-sm">
                {(cols.data ?? []).map((c) => <li key={c.id} className="rounded bg-surface-2 px-2.5 py-1">{c.name}</li>)}
                {cols.data?.length === 0 && <li className="text-dim">No collections yet.</li>}
              </ul>
              <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); addCol.mutate(); }}>
                <Input aria-label="Collection name" placeholder="Collection name" value={colName} onChange={(e) => setColName(e.target.value)} required />
                <Button type="submit" loading={addCol.isPending}>Add</Button>
              </form>
              <ErrorBox error={addCol.error} />
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
