import { useCallback, useEffect, useState } from "react";
import { ApiError, listDocuments } from "./api";
import type { DocumentInfo } from "./api";
import { logout, me } from "./auth";
import type { User } from "./auth";
import AnalysisView from "./components/AnalysisView";
import Chat from "./components/Chat";
import DocumentPanel from "./components/DocumentPanel";
import EvalView from "./components/EvalView";
import LoginView from "./components/LoginView";
import { clearToken, getToken, setUnauthorizedHandler } from "./session";
import "./App.css";
import "./Analysis.css";

type View = "chat" | "analyze" | "eval";

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(true);
  const [startupError, setStartupError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [view, setView] = useState<View>("chat");

  // Back to the sign-in screen. Everything user-specific is dropped, and the
  // signed-in views unmount, which also clears the chat.
  const resetSession = useCallback(() => {
    clearToken();
    setUser(null);
    setDocuments([]);
    setSelected([]);
    setLoadError(null);
    setView("chat");
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(resetSession);
    return () => setUnauthorizedHandler(null);
  }, [resetSession]);

  // Restore a saved session on first load.
  useEffect(() => {
    if (!getToken()) {
      setChecking(false);
      return;
    }
    me()
      .then(setUser)
      .catch((err: unknown) => {
        // A 401 already cleared the token. Other failures keep it, so a reload can retry.
        if (err instanceof ApiError && err.status !== 401) setStartupError(err.message);
      })
      .finally(() => setChecking(false));
  }, []);

  const refresh = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      // Drop selections for documents that no longer exist.
      setSelected((prev) => prev.filter((id) => docs.some((d) => d.id === id)));
      setLoadError(null);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Unknown error");
    }
  }, []);

  useEffect(() => {
    if (user) void refresh();
  }, [user, refresh]);

  async function signOut() {
    try {
      await logout();
    } catch {
      // The session may already be gone; we sign out locally either way.
    }
    resetSession();
  }

  function toggle(id: string) {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  if (checking) {
    return (
      <div className="auth">
        <p className="muted">Loading…</p>
      </div>
    );
  }
  if (!user) {
    return <LoginView onAuthenticated={setUser} notice={startupError} />;
  }

  const readyCount = documents.filter((d) => d.status === "ready").length;

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>ResearchIQ</h1>
        <p className="muted small">Answers grounded in your documents</p>
        <p className="account small">
          <span className="muted">{user.email}</span>
          <button className="link-btn" onClick={() => void signOut()}>
            Sign out
          </button>
        </p>
        {loadError && (
          <div className="banner bad small">
            {loadError}{" "}
            <button className="link-btn" onClick={() => void refresh()}>
              Retry
            </button>
          </div>
        )}
        <DocumentPanel documents={documents} selected={selected} onToggle={toggle} onChanged={refresh} />
      </aside>
      <main className="main">
        <nav className="tabs">
          <button className={"tab" + (view === "chat" ? " active" : "")} onClick={() => setView("chat")}>
            Chat
          </button>
          <button
            className={"tab" + (view === "analyze" ? " active" : "")}
            onClick={() => setView("analyze")}
          >
            Summarize &amp; compare
          </button>
          {user.is_admin && (
            <button className={"tab" + (view === "eval" ? " active" : "")} onClick={() => setView("eval")}>
              Evaluation
            </button>
          )}
        </nav>
        {/* All views stay mounted so switching tabs doesn't lose state. */}
        <div className="view" hidden={view !== "chat"}>
          <Chat selectedIds={selected} readyCount={readyCount} />
        </div>
        <div className="view" hidden={view !== "analyze"}>
          <AnalysisView documents={documents} />
        </div>
        {user.is_admin && (
          <div className="view" hidden={view !== "eval"}>
            <EvalView active={view === "eval"} />
          </div>
        )}
      </main>
    </div>
  );
}