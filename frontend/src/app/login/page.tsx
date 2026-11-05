"use client";

import { FlaskConical } from "lucide-react";
import { useState } from "react";
import { Button, ErrorBox, Field, Input } from "@/components/ui";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "login") await login(email, password);
      else await register(email, password, name);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-screen place-items-center p-4">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex items-center gap-3">
          <div className="grid size-10 place-items-center rounded-lg bg-accent text-accent-ink"><FlaskConical className="size-5" aria-hidden /></div>
          <div>
            <h1 className="text-lg font-semibold leading-tight">RAG</h1>
            <p className="text-xs text-dim">Build, compare and understand retrieval.</p>
          </div>
        </div>
        <form onSubmit={submit} className="space-y-3 rounded-lg border border-border bg-surface p-5">
          <div role="tablist" className="grid grid-cols-2 gap-1 rounded-md bg-surface-2 p-1 text-sm">
            {(["login", "register"] as const).map((m) => (
              <button key={m} type="button" role="tab" aria-selected={mode === m} onClick={() => setMode(m)}
                className={`rounded px-2 py-1 ${mode === m ? "bg-surface font-medium shadow-sm" : "text-dim"}`}>
                {m === "login" ? "Sign in" : "Create account"}
              </button>
            ))}
          </div>
          {mode === "register" && (
            <Field label="Your name"><Input value={name} onChange={(e) => setName(e.target.value)} required autoComplete="name" /></Field>
          )}
          <Field label="Email"><Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" /></Field>
          <Field label="Password" hint={mode === "register" ? "At least 8 characters." : undefined}>
            <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={mode === "register" ? 8 : undefined}
              autoComplete={mode === "login" ? "current-password" : "new-password"} />
          </Field>
          <ErrorBox error={error} />
          <Button type="submit" variant="primary" loading={busy} className="w-full">{mode === "login" ? "Sign in" : "Create account"}</Button>
        </form>
      </div>
    </div>
  );
}
