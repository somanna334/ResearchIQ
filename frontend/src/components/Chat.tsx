import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { ask } from "../api";
import type { AskResponse, Mode } from "../api";
import AnswerView from "./AnswerView";

interface Turn {
  id: number;
  question: string;
  state: "loading" | "done" | "error";
  response?: AskResponse;
  error?: string;
}

interface Props {
  selectedIds: string[];
  readyCount: number;
}

export default function Chat({ selectedIds, readyCount }: Props) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("semantic");
  const [k, setK] = useState(5);
  const [pending, setPending] = useState(false);
  const nextId = useRef(1);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  async function send() {
    const question = input.trim();
    if (!question || pending) return;
    const id = nextId.current++;
    setTurns((prev) => [...prev, { id, question, state: "loading" }]);
    setInput("");
    setPending(true);
    try {
      const response = await ask({
        question,
        k,
        mode,
        document_ids: selectedIds.length > 0 ? selectedIds : undefined,
      });
      setTurns((prev) =>
        prev.map((t): Turn => (t.id === id ? { ...t, state: "done", response } : t)),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed";
      setTurns((prev) =>
        prev.map((t): Turn => (t.id === id ? { ...t, state: "error", error: message } : t)),
      );
    } finally {
      setPending(false);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void send();
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  }

  const scope =
    selectedIds.length > 0
      ? `Searching ${selectedIds.length} selected document${selectedIds.length === 1 ? "" : "s"}`
      : readyCount > 0
        ? "Searching all documents"
        : "Upload a PDF to get started";

  return (
    <section className="chat">
      <div className="chat-log">
        {turns.length === 0 && (
          <div className="empty">
            <h2>Ask your documents</h2>
            <p className="muted">
              Answers come only from the documents you upload, with a citation for every claim. If
              the evidence isn't there, the app says so instead of guessing.
            </p>
          </div>
        )}
        {turns.map((turn) => (
          <div key={turn.id} className="turn">
            <div className="bubble-q">{turn.question}</div>
            {turn.state === "loading" && <div className="card muted">Thinking…</div>}
            {turn.state === "error" && <div className="card banner bad">{turn.error}</div>}
            {turn.state === "done" && turn.response && <AnswerView response={turn.response} />}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form className="composer" onSubmit={onSubmit}>
        <div className="composer-opts">
          <span className="muted small">{scope}</span>
          <label className="small">
            Retrieval{" "}
            <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              <option value="semantic">Semantic (fast baseline)</option>
              <option value="hybrid">Hybrid (keyword + vector)</option>
              <option value="hybrid_rerank">Hybrid + rerank (slower)</option>
            </select>
          </label>
          <label className="small">
            Passages{" "}
            <select value={k} onChange={(e) => setK(Number(e.target.value))}>
              <option value={3}>3</option>
              <option value={5}>5</option>
              <option value={8}>8</option>
            </select>
          </label>
        </div>
        {mode === "hybrid_rerank" && (
          <p className="muted small">The first reranked request loads a model, so it can be slow.</p>
        )}
        <div className="composer-row">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a question about your documents…"
            rows={2}
            maxLength={1000}
            disabled={readyCount === 0}
          />
          <button type="submit" className="primary" disabled={pending || !input.trim() || readyCount === 0}>
            {pending ? "…" : "Ask"}
          </button>
        </div>
      </form>
    </section>
  );
}