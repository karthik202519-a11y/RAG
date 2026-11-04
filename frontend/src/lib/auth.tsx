"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, getToken, setToken, type User } from "./api";

type AuthState = {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => void;
};

const Ctx = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  // Loading only while a stored token is being verified; no token means we know immediately.
  const [loading, setLoading] = useState(() => typeof window !== "undefined" && !!getToken());

  useEffect(() => {
    const onLogout = () => setUser(null);
    window.addEventListener("studio:logout", onLogout);
    if (getToken()) {
      api<User>("/auth/me")
        .then(setUser)
        .catch(() => setToken(null))
        .finally(() => setLoading(false));
    }
    return () => window.removeEventListener("studio:logout", onLogout);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const r = await api<{ token: string; user: User }>("/auth/login", { method: "POST", json: { email, password } });
    setToken(r.token);
    setUser(r.user);
  }, []);
  const register = useCallback(async (email: string, password: string, display_name: string) => {
    const r = await api<{ token: string; user: User }>("/auth/register", {
      method: "POST",
      json: { email, password, display_name },
    });
    setToken(r.token);
    setUser(r.user);
  }, []);
  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
  }, []);

  const value = useMemo(() => ({ user, loading, login, register, logout }), [user, loading, login, register, logout]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAuth(): AuthState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAuth outside AuthProvider");
  return v;
}
