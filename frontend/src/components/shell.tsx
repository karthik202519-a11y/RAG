"use client";

import clsx from "clsx";
import {
  BookOpen, ClipboardCheck, Database, FileStack, FolderKanban, LayoutDashboard, LogOut, Menu, Moon, Activity, Network, Settings, ScanText, Scissors, ScatterChart, Search, Sun, X,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, type Catalog, type EvalDatasetInfo, type IndexInfo, type Providers } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PLANNED } from "@/lib/planned";
import { useProject } from "@/lib/project";
import { Spinner } from "./ui";

// Grouped by the order people actually work: set up, build the pipeline, look at it, measure it.
const GROUPS = [
  { label: "Overview", items: [
    { href: "/", label: "Dashboard", icon: LayoutDashboard },
    { href: "/projects", label: "Projects & Collections", icon: FolderKanban } ] },
  { label: "Build", items: [
    { href: "/ingestion", label: "Document Ingestion", icon: FileStack },
    { href: "/parsing", label: "Parser Lab", icon: ScanText },
    { href: "/chunking", label: "Chunking Playground", icon: Scissors },
    { href: "/indexes", label: "Indexes & Embeddings", icon: Database } ] },
  { label: "Explore", items: [
    { href: "/retrieval", label: "Retrieval Lab", icon: Search },
    { href: "/embeddings", label: "Embedding Explorer", icon: ScatterChart },
    { href: "/graph", label: "GraphRAG Explorer", icon: Network } ] },
  { label: "Measure", items: [
    { href: "/evaluation", label: "Evaluation Center", icon: ClipboardCheck },
    { href: "/traces", label: "Traces", icon: Activity } ] },
  { label: "Learn", items: [{ href: "/catalog", label: "Method Catalog", icon: BookOpen }] },
  { label: "System", items: [{ href: "/settings", label: "Settings", icon: Settings }] },
];

function initialTheme(): "light" | "dark" {
  if (typeof window === "undefined") return "light";
  let t: string | null = null;
  try {
    t = window.localStorage.getItem("studio.theme");
  } catch {
    /* ignore */
  }
  if (t === "light" || t === "dark") return t;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(initialTheme);
  useEffect(() => {
    document.documentElement.dataset.theme = theme; // sync the DOM with state; no state is set here
  }, [theme]);
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    try {
      window.localStorage.setItem("studio.theme", next);
    } catch {
      /* ignore */
    }
  };
  return { theme, toggle };
}

/** Three staggered bars: text cut into chunks, the thing this product is about. */
function BrandMark() {
  return (
    <div className="grid size-8 shrink-0 place-items-center rounded-md bg-accent" aria-hidden>
      <svg viewBox="0 0 20 20" className="size-5" fill="none">
        <rect x="2" y="3.5" width="11" height="3" rx="1.5" fill="var(--accent-ink)" />
        <rect x="7" y="8.5" width="11" height="3" rx="1.5" fill="var(--accent-ink)" opacity="0.7" />
        <rect x="4" y="13.5" width="9" height="3" rx="1.5" fill="var(--accent-ink)" opacity="0.45" />
      </svg>
    </div>
  );
}

/** Warm the cache for a screen as soon as someone reaches for its link, so it opens with data already there. */
function usePrefetch(projectId: string | undefined) {
  const qc = useQueryClient();
  return (href: string) => {
    const pre = (key: unknown[], fn: () => Promise<unknown>, staleTime = 30_000) => void qc.prefetchQuery({ queryKey: key, queryFn: fn, staleTime });
    if (href === "/catalog") pre(["catalog"], () => api<Catalog>("/method-catalog"), Infinity);
    if (href === "/chunking" || href === "/indexes") pre(["catalog", "chunking"], () => api<Catalog>("/method-catalog?category=chunking"), Infinity);
    if (href === "/indexes" || href === "/retrieval") pre(["providers"], () => api<Providers>("/providers"), 60_000);
    if (!projectId) return;
    if (href === "/retrieval" || href === "/embeddings" || href === "/evaluation" || href === "/indexes") pre(["indexes", projectId], () => api<IndexInfo[]>(`/projects/${projectId}/indexes`));
    if (href === "/evaluation") pre(["eval-datasets", projectId], () => api<EvalDatasetInfo[]>(`/projects/${projectId}/eval-datasets`));
    if (href === "/chunking" || href === "/indexes" || href === "/retrieval") pre(["docs-ready", projectId], () => api(`/projects/${projectId}/documents?status=ready&limit=500`));
  };
}

function NavLinks({ pathname, onNavigate, onIntent }: { pathname: string; onNavigate?: () => void; onIntent?: (href: string) => void }) {
  return (
    <nav aria-label="Main" className="space-y-4 px-2 pb-4">
      {GROUPS.map((g) => (
        <div key={g.label}>
          <div className="px-2.5 pb-1 text-xs font-medium text-dim">{g.label}</div>
          {g.items.map((n) => {
            const active = n.href === "/" ? pathname === "/" : pathname.startsWith(n.href);
            return (
              <Link key={n.href} href={n.href} onClick={onNavigate} onMouseEnter={() => onIntent?.(n.href)} onFocus={() => onIntent?.(n.href)} aria-current={active ? "page" : undefined}
                className={clsx("flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm hover:bg-surface-2",
                  active && "bg-accent-soft font-medium text-accent")}>
                <n.icon className="size-4 shrink-0" aria-hidden /> {n.label}
              </Link>
            );
          })}
        </div>
      ))}
      <details className="group" open={pathname.startsWith("/planned")}>
        <summary className="flex cursor-pointer list-none items-center justify-between rounded-md px-2.5 py-1.5 text-xs font-medium text-dim hover:bg-surface-2">
          <span>Not built yet ({PLANNED.length})</span>
          <span aria-hidden className="transition-transform group-open:rotate-90">›</span>
        </summary>
        <div className="mt-1">
          {PLANNED.map((p) => {
            const href = `/planned/${p.slug}`;
            return (
              <Link key={p.slug} href={href} onClick={onNavigate} aria-current={pathname === href ? "page" : undefined}
                className={clsx("block rounded-md px-2.5 py-1 text-[13px] text-dim hover:bg-surface-2", pathname === href && "bg-surface-2 text-text")}>
                {p.title}
              </Link>
            );
          })}
        </div>
      </details>
    </nav>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const { user, loading, logout } = useAuth();
  const { projects, project, setProjectId, listLoading: projectsLoading } = useProject();
  const pathname = usePathname();
  const router = useRouter();
  const { theme, toggle } = useTheme();
  const prefetch = usePrefetch(project?.id);
  const [menu, setMenu] = useState(false);
  const isLogin = pathname === "/login";

  useEffect(() => {
    if (!loading && !user && !isLogin) router.replace("/login");
    if (!loading && user && isLogin) router.replace("/");
  }, [loading, user, isLogin, router]);

  useEffect(() => {
    if (!menu) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setMenu(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menu]);

  if (isLogin) return <>{children}</>;
  if (loading || !user) return <Spinner label="Checking your session" />;

  return (
    <div className="flex min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2 focus:shadow">Skip to content</a>
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col overflow-y-auto border-r border-border bg-surface md:flex">
        <div className="flex items-center gap-2.5 px-4 py-4">
          <BrandMark />
          <div className="leading-tight">
            <div className="text-sm font-semibold">RAG Learning Studio</div>
            <div className="text-xs text-dim">retrieval workbench</div>
          </div>
        </div>
        <NavLinks pathname={pathname} onIntent={prefetch} />
      </aside>

      {menu && (
        <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <button className="absolute inset-0 bg-black/40" aria-label="Close navigation" onClick={() => setMenu(false)} />
          <div className="absolute inset-y-0 left-0 w-72 max-w-[85vw] overflow-y-auto bg-surface shadow-xl">
            <div className="flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-2.5"><BrandMark /><span className="text-sm font-semibold">RAG Learning Studio</span></div>
              <button onClick={() => setMenu(false)} aria-label="Close navigation" className="grid size-8 place-items-center rounded-md hover:bg-surface-2"><X className="size-4" /></button>
            </div>
            <NavLinks pathname={pathname} onNavigate={() => setMenu(false)} onIntent={prefetch} />
          </div>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-border bg-bg/90 px-4 py-2 backdrop-blur">
          <button onClick={() => setMenu(true)} aria-label="Open navigation" aria-expanded={menu}
            className="grid size-8 place-items-center rounded-md border border-border bg-surface hover:bg-surface-2 md:hidden"><Menu className="size-4" /></button>
          <label className="flex min-w-0 items-center gap-2 text-xs text-dim">
            <span className="hidden sm:inline">Project</span>
            <select aria-label="Project" value={project?.id ?? ""} disabled={projectsLoading} onChange={(e) => setProjectId(e.target.value)}
              className="h-8 max-w-[11rem] truncate rounded-md border border-border bg-surface px-2 text-sm text-text disabled:text-dim sm:max-w-xs">
              {projectsLoading && <option value="">Loading projects…</option>}
              {!projectsLoading && projects.length === 0 && <option value="">No projects yet</option>}
              {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </label>
          <div className="ml-auto flex items-center gap-2">
            <span className="hidden text-xs text-dim sm:inline">{user.display_name}</span>
            <button onClick={toggle} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              className="grid size-8 place-items-center rounded-md border border-border bg-surface hover:bg-surface-2">
              {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
            <button onClick={() => { logout(); router.replace("/login"); }} aria-label="Sign out"
              className="grid size-8 place-items-center rounded-md border border-border bg-surface hover:bg-surface-2">
              <LogOut className="size-4" />
            </button>
          </div>
        </header>
        <main id="main" className="flex-1 p-4 sm:p-5">{children}</main>
      </div>
    </div>
  );
}
