"use client";

import { useQuery } from "@tanstack/react-query";
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { api, type Project } from "./api";
import { useAuth } from "./auth";

type Ctx = {
  projects: Project[];
  project: Project | null;
  setProjectId: (id: string) => void;
  loading: boolean;      // no project known yet
  listLoading: boolean;  // the project list is still on its way
  refetch: () => void;
};
const C = createContext<Ctx | null>(null);
const KEY = "studio.project";
const OBJ = "studio.projectobj";

/** The last project we showed. Lets every page start its own requests at once instead of waiting for the list. */
function remembered(): Project | null {
  if (typeof window === "undefined") return null;
  try {
    return JSON.parse(window.localStorage.getItem(OBJ) ?? "null") as Project | null;
  } catch {
    return null;
  }
}

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

  const [cached] = useState<Project | null>(remembered);
  const projects = q.data ?? [];
  const fromList = projects.find((p) => p.id === id) ?? projects[0] ?? null;
  // While the list loads, use the remembered project (if it is the one selected) so queries are not delayed.
  const project = fromList ?? (q.isLoading && cached && (!id || cached.id === id) ? cached : null);
  useEffect(() => {
    if (!fromList) return;
    try {
      window.localStorage.setItem(OBJ, JSON.stringify(fromList));
    } catch {
      /* ignore */
    }
  }, [fromList]);

  const value = useMemo<Ctx>(
    () => ({
      projects,
      project,
      loading: q.isLoading && !project,
      listLoading: q.isLoading,
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
