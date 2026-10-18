import { useEffect, useId, useState } from "react";
import type { AskResponse, Source } from "../api";
import { scoreLabel, splitCitations } from "../lib";
import PageModal from "./PageModal";

interface Props {
  response: AskResponse;
}

export default function AnswerView({ response }: Props) {
  const baseId = useId();
  const [open, setOpen] = useState(false);
  const [focus, setFocus] = useState<string | null>(null);
  const [pageSource, setPageSource] = useState<Source | null>(null);

  const { sources, usage, retrieval, status } = response;
  const byTag = new Map(sources.map((s) => [s.tag, s]));

  useEffect(() => {
    if (focus) {
      document
        .getElementById(`${baseId}-${focus}`)
        ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [focus, open, baseId]);

  function jumpTo(tag: string) {
    setOpen(true);
    setFocus(tag);
  }

  function renderAnswer() {
    if (status !== "answered") return <p className="muted">{response.answer}</p>;
    return (
      <p className="answer-text">
        {splitCitations(response.answer).map((segment, i) =>
          segment.kind === "text" ? (
            <span key={i}>{segment.text}</span>
          ) : (
            <span key={i} className="chips">
              {segment.tags.map((tag) => {
                const source = byTag.get(tag);
                return source ? (
                  <button
                    key={tag}
                    className="chip"
                    onClick={() => jumpTo(tag)}
                    title={`${source.filename}, page ${source.page_number}`}
                  >
                    {tag} · p.{source.page_number}
                  </button>
                ) : (
                  <span key={tag} className="chip invalid">
                    {tag}
                  </span>
                );
              })}
            </span>
          ),
        )}
      </p>
    );
  }

  return (
    <div className="card">
      {status === "insufficient_evidence" && retrieval.gated && (
        <div className="banner warn">
          <strong>Not answered.</strong> The best matching passage was too weak (similarity{" "}
          {retrieval.best_dense_score?.toFixed(3)}, threshold {retrieval.min_top_score}), so the
          model was not called.
        </div>
      )}
      {status === "insufficient_evidence" && !retrieval.gated && sources.length > 0 && (
        <div className="banner warn">
          <strong>Not answered.</strong> The model judged that the retrieved passages don't contain
          the answer.
        </div>
      )}
      {status === "insufficient_evidence" && sources.length === 0 && (
        <div className="banner warn">
          <strong>Nothing to search.</strong> No indexed passages were found. Upload a PDF, or use
          Re-index on a document.
        </div>
      )}
      {status === "citation_error" && (
        <div className="banner bad">
          <strong>Answer withheld.</strong> Its citations couldn't be verified against the
          retrieved passages.
        </div>
      )}

      {renderAnswer()}

      <p className="usage muted small">
        {usage.model} · {retrieval.mode} · retrieval {usage.retrieval_ms} ms · model{" "}
        {usage.llm_ms} ms · {usage.prompt_tokens ?? "–"} in / {usage.completion_tokens ?? "–"} out
        tokens
      </p>

      {sources.length > 0 && (
        <details open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
          <summary>Retrieved passages ({sources.length})</summary>
          <p className="muted small">
            Scores use a different scale in each retrieval mode, so don't compare them across modes.
          </p>
          <ol className="passages">
            {sources.map((source) => (
              <li
                key={source.tag}
                id={`${baseId}-${source.tag}`}
                className={
                  "passage" + (source.cited ? " cited" : "") + (focus === source.tag ? " focus" : "")
                }
              >
                <div className="passage-head">
                  <span>
                    <strong>{source.tag}</strong> · {source.filename} · page {source.page_number}
                  </span>
                  <span className="muted small">
                    {scoreLabel(retrieval.mode, source.score)}
                    {source.cited && " · cited"}
                  </span>
                </div>
                <p className="passage-text">{source.text}</p>
                <button className="link-btn" onClick={() => setPageSource(source)}>
                  View full page
                </button>
              </li>
            ))}
          </ol>
        </details>
      )}

      {pageSource && <PageModal source={pageSource} onClose={() => setPageSource(null)} />}
    </div>
  );
}