import { useState } from "react";
import type { DocumentInfo, Mode } from "../api";
import { compareDocuments, summarizeDocument } from "../analysis";
import type { CompareResponse, SummaryResponse } from "../analysis";
import { toggleLimited } from "../selection";
import AnalysisResult from "./AnalysisResult";
import "../Analysis.css";

type Loadable<T> =
  | { state: "idle" }
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "done"; data: T };

const MAX_COMPARE = 3;

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : "Request failed";
}

export default function AnalysisView({ documents }: { documents: DocumentInfo[] }) {
  const ready = documents.filter((d) => d.status === "ready");

  const [summaryId, setSummaryId] = useState("");
  const [summary, setSummary] = useState<Loadable<SummaryResponse>>({ state: "idle" });

  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("semantic");
  const [comparison, setComparison] = useState<Loadable<CompareResponse>>({ state: "idle" });

  // A stale selection (for example a deleted document) falls back to the first ready one.
  const summaryTarget = ready.some((d) => d.id === summaryId) ? summaryId : (ready[0]?.id ?? "");
  const picked = compareIds.filter((id) => ready.some((d) => d.id === id));

  async function runSummary() {
    if (!summaryTarget) return;
    setSummary({ state: "loading" });
    try {
      setSummary({ state: "done", data: await summarizeDocument(summaryTarget) });
    } catch (err) {
      setSummary({ state: "error", message: messageOf(err) });
    }
  }

  async function runCompare() {
    if (picked.length < 2 || !question.trim()) return;
    setComparison({ state: "loading" });
    try {
      const data = await compareDocuments({
        document_ids: picked,
        question: question.trim(),
        k: 4,
        mode,
      });
      setComparison({ state: "done", data });
    } catch (err) {
      setComparison({ state: "error", message: messageOf(err) });
    }
  }

  const s = summary.state === "done" ? summary.data : null;
  const c = comparison.state === "done" ? comparison.data : null;
  const labels = c
    ? Object.fromEntries(c.documents.map((d) => [d.document_id, d.label]))
    : undefined;
  const uncited = c ? c.documents.filter((d) => !d.cited) : [];

  return (
    <div className="analysis">
      <div className="analysis-inner">
        {ready.length === 0 && (
          <div className="banner warn">Upload at least one PDF to use these tools.</div>
        )}

        <section className="panel">
          <h2>Summarize a document</h2>
          <p className="muted small">
            Every bullet cites its passages. Very long documents are sampled evenly, and the result
            tells you how much was covered.
          </p>
          <div className="row">
            <select
              value={summaryTarget}
              onChange={(e) => setSummaryId(e.target.value)}
              disabled={ready.length === 0}
            >
              {ready.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.filename}
                </option>
              ))}
            </select>
            <button
              className="primary"
              disabled={!summaryTarget || summary.state === "loading"}
              onClick={() => void runSummary()}
            >
              {summary.state === "loading" ? "Summarizing…" : "Summarize"}
            </button>
          </div>

          {summary.state === "error" && <div className="banner bad">{summary.message}</div>}
          {s && (
            <>
              {s.status === "citation_error" && (
                <div className="banner bad">
                  <strong>Summary withheld.</strong> Its citations couldn't be verified.
                </div>
              )}
              {s.status === "insufficient_evidence" && (
                <div className="banner warn">The model found no meaningful content to summarize.</div>
              )}
              <p className="muted small">
                Based on {s.coverage.chunks_used} of {s.coverage.chunks_total} passages, covering{" "}
                {s.coverage.pages_covered.length} of {s.coverage.page_count} pages.
              </p>
              {s.coverage.sampled && (
                <div className="banner warn small">
                  This document is longer than the summary budget, so passages were sampled evenly.
                  Parts of it are not represented.
                </div>
              )}
              <AnalysisResult
                text={s.summary}
                validated={s.status === "answered"}
                sources={s.sources}
              />
              <p className="muted small">
                {s.usage.model} · {s.usage.llm_ms} ms · {s.usage.prompt_tokens ?? "–"} in /{" "}
                {s.usage.completion_tokens ?? "–"} out tokens
              </p>
            </>
          )}
        </section>

        <section className="panel">
          <h2>Compare documents</h2>
          <p className="muted small">
            Pick 2 or {MAX_COMPARE} documents and a question. Evidence is retrieved from each one
            separately, so every document contributes.
          </p>
          <div className="doc-pick">
            {ready.map((d) => (
              <label key={d.id}>
                <input
                  type="checkbox"
                  checked={picked.includes(d.id)}
                  disabled={!picked.includes(d.id) && picked.length >= MAX_COMPARE}
                  onChange={() => setCompareIds(toggleLimited(picked, d.id, MAX_COMPARE))}
                />
                {d.filename}
              </label>
            ))}
          </div>
          <textarea
            rows={2}
            maxLength={1000}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What should be compared? For example: How do their admission requirements differ?"
          />
          <div className="row">
            <label className="small">
              Retrieval{" "}
              <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
                <option value="semantic">Semantic (fast baseline)</option>
                <option value="hybrid">Hybrid</option>
                <option value="hybrid_rerank">Hybrid + rerank (slower)</option>
              </select>
            </label>
            <button
              className="primary"
              disabled={picked.length < 2 || !question.trim() || comparison.state === "loading"}
              onClick={() => void runCompare()}
            >
              {comparison.state === "loading" ? "Comparing…" : "Compare"}
            </button>
          </div>

          {comparison.state === "error" && <div className="banner bad">{comparison.message}</div>}
          {c && (
            <>
              {c.status === "insufficient_evidence" && c.retrieval.gated && (
                <div className="banner warn">
                  <strong>Not compared.</strong> None of the selected documents had a passage
                  similar enough to the question (threshold {c.retrieval.min_top_score}), so the
                  model was not called.
                </div>
              )}
              {c.status === "insufficient_evidence" && !c.retrieval.gated && (
                <div className="banner warn">
                  <strong>Not answered.</strong> The model judged that the retrieved passages don't
                  address the question.
                </div>
              )}
              {c.status === "citation_error" && (
                <div className="banner bad">
                  <strong>Answer withheld.</strong> Its citations couldn't be verified.
                </div>
              )}
              {c.status === "answered" && uncited.length > 0 && (
                <div className="banner warn small">
                  The answer doesn't cite {uncited.map((d) => `Document ${d.label}`).join(", ")}.
                  Treat any claims about it with caution, and check its passages below.
                </div>
              )}
              <ul className="coverage-list small">
                {c.documents.map((d) => (
                  <li key={d.document_id}>
                    <strong>Document {d.label}</strong> · {d.filename} · {d.passages} passages
                    {d.best_dense_score !== null && ` · best similarity ${d.best_dense_score.toFixed(3)}`}
                    {c.status === "answered" && (d.cited ? " · cited" : " · not cited")}
                  </li>
                ))}
              </ul>
              <AnalysisResult
                text={c.answer}
                validated={c.status === "answered"}
                sources={c.sources}
                mode={c.retrieval.mode}
                labels={labels}
              />
              <p className="muted small">
                {c.usage.model} · retrieval {c.usage.retrieval_ms} ms · model {c.usage.llm_ms} ms ·{" "}
                {c.usage.prompt_tokens ?? "–"} in / {c.usage.completion_tokens ?? "–"} out tokens
              </p>
            </>
          )}
        </section>
      </div>
    </div>
  );
}