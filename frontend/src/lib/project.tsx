"use client";

import { useQuery } from "@tanstack/react-query";
import { createContext, useContext, useMemo, useState } from "react";
import { api, type Project } from "./api";
import { useAuth } from "./auth";

type Ctx = {
  projects: Project[];
  project: Project | null;
  setProjectId: (id: string) => void;
  loading: boolean;
  refetch: () => void;
};
const C = createContext<Ctx | null>(null);
const KEY = "studio.project";

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const q = useQuery({ queryKey: ["projects"], queryFn: () => api<Project[]>("/projects"), enabled: !!user });
  const [id, setId] = useState<string | null>(() => {
    if (typeof window === "undefined") return null;
    try {
      return window.localStorage.getItem(KEY);
    } catch {
      return null;
    }
  });

  const projects = q.data ?? [];
  const project = projects.find((p) => p.id === id) ?? projects[0] ?? null;

  const value = useMemo<Ctx>(
    () => ({
      projects,
      project,
      loading: q.isLoading,
      refetch: () => void q.refetch(),
      setProjectId: (pid: string) => {
        setId(pid);
        try {
          window.localStorage.setItem(KEY, pid);
        } catch {
          /* ignore */
        }
      },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [projects, project, q.isLoading],
  );
  return <C.Provider value={value}>{children}</C.Provider>;
}

export function useProject(): Ctx {
  const v = useContext(C);
  if (!v) throw new Error("useProject outside ProjectProvider");
  return v;
}
