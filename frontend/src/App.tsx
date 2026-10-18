import { useCallback, useEffect, useState } from "react";
import { listDocuments } from "./api";
import type { DocumentInfo } from "./api";
import Chat from "./components/Chat";
import DocumentPanel from "./components/DocumentPanel";
import "./App.css";

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

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
    void refresh();
  }, [refresh]);

  function toggle(id: string) {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  const readyCount = documents.filter((d) => d.status === "ready").length;

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>ResearchIQ</h1>
        <p className="muted small">Answers grounded in your documents</p>
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
        <Chat selectedIds={selected} readyCount={readyCount} />
      </main>
    </div>
  );
}