"use client";

import clsx from "clsx";
import {
  BookOpen, Boxes, Database, FileStack, FlaskConical, FolderKanban, LayoutDashboard, LogOut, Moon, Scissors, Search, Sun, Wrench,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { PLANNED } from "@/lib/planned";
import { useProject } from "@/lib/project";
import { Spinner } from "./ui";

const BUILT = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/projects", label: "Projects & Collections", icon: FolderKanban },
  { href: "/ingestion", label: "Document Ingestion", icon: FileStack },
  { href: "/chunking", label: "Chunking Playground", icon: Scissors },
  { href: "/indexes", label: "Indexes & Embeddings", icon: Database },
  { href: "/retrieval", label: "Retrieval Lab", icon: Search },
  { href: "/catalog", label: "Method Catalog", icon: BookOpen },
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

export function AppShell({ children }: { children: React.ReactNode }) {
  const { user, loading, logout } = useAuth();
  const { projects, project, setProjectId } = useProject();
  const pathname = usePathname();
  const router = useRouter();
  const { theme, toggle } = useTheme();
  const isLogin = pathname === "/login";

  useEffect(() => {
    if (!loading && !user && !isLogin) router.replace("/login");
    if (!loading && user && isLogin) router.replace("/");
  }, [loading, user, isLogin, router]);

  if (isLogin) return <>{children}</>;
  if (loading || !user) return <Spinner label="Checking your session" />;

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col overflow-y-auto border-r border-border bg-surface md:flex">
        <div className="flex items-center gap-2 px-4 py-4">
          <div className="grid size-8 place-items-center rounded-md bg-accent text-accent-ink"><FlaskConical className="size-4" aria-hidden /></div>
          <div className="leading-tight">
            <div className="text-sm font-semibold">RAG Learning Studio</div>
            <div className="text-[11px] text-dim">engineering laboratory</div>
          </div>
        </div>
        <nav aria-label="Main" className="flex-1 space-y-5 px-2 pb-4">
          <div>
            <div className="px-2 pb-1 text-[11px] font-semibold uppercase tracking-wide text-dim">Working</div>
            {BUILT.map((n) => {
              const active = n.href === "/" ? pathname === "/" : pathname.startsWith(n.href);
              return (
                <Link key={n.href} href={n.href} aria-current={active ? "page" : undefined}
                  className={clsx("flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm hover:bg-surface-2",
                    active && "bg-accent-soft font-medium text-accent")}>
                  <n.icon className="size-4" aria-hidden /> {n.label}
                </Link>
              );
            })}
          </div>
          <div>
            <div className="flex items-center gap-1.5 px-2 pb-1 text-[11px] font-semibold uppercase tracking-wide text-dim">
              <Wrench className="size-3" aria-hidden /> Not built yet
            </div>
            {PLANNED.map((p) => {
              const href = `/planned/${p.slug}`;
              const active = pathname === href;
              return (
                <Link key={p.slug} href={href} aria-current={active ? "page" : undefined}
                  className={clsx("flex items-center justify-between rounded-md px-2.5 py-1.5 text-[13px] text-dim hover:bg-surface-2",
                    active && "bg-surface-2 text-text")}>
                  <span>{p.title}</span>
                </Link>
              );
            })}
          </div>
        </nav>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-border bg-bg/90 px-4 py-2 backdrop-blur">
          <Boxes className="size-4 text-dim" aria-hidden />
          <label className="flex items-center gap-2 text-xs text-dim">
            Project
            <select aria-label="Project" value={project?.id ?? ""} onChange={(e) => setProjectId(e.target.value)}
              className="h-8 rounded-md border border-border bg-surface px-2 text-sm text-text">
              {projects.length === 0 && <option value="">No projects yet</option>}
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
        <main className="flex-1 p-5">{children}</main>
      </div>
    </div>
  );
}
