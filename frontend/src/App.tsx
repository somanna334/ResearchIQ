import { useEffect, useState } from "react";
import "./App.css";

const API_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type Health = { status: string };

type State =
  | { kind: "loading" }
  | { kind: "ok"; data: Health }
  | { kind: "error"; message: string };

function App() {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${API_URL}/health`, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<Health>;
      })
      .then((data) => setState({ kind: "ok", data }))
      .catch((err: Error) => {
        if (err.name !== "AbortError") {
          setState({ kind: "error", message: err.message });
        }
      });

    return () => controller.abort();
  }, []);

  return (
    <main className="app">
      <h1>ResearchIQ</h1>
      <p>
        Backend status:{" "}
        {state.kind === "loading" && <span>checking…</span>}
        {state.kind === "ok" && <strong className="ok">{state.data.status}</strong>}
        {state.kind === "error" && (
          <strong className="bad">unreachable ({state.message})</strong>
        )}
      </p>
    </main>
  );
}

export default App;