"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { AuthProvider } from "@/lib/auth";
import { ProjectProvider } from "@/lib/project";
import { AppShell } from "./shell";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () => new QueryClient({ defaultOptions: { queries: { staleTime: 30_000, gcTime: 10 * 60_000, retry: 1, refetchOnWindowFocus: false } } }),
  );
  return (
    <QueryClientProvider client={client}>
      <AuthProvider>
        <ProjectProvider>
          <AppShell>{children}</AppShell>
        </ProjectProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
}
