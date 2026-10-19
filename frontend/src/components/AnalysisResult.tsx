import { useEffect, useId, useState } from "react";
import type { Mode, Source } from "../api";
import { scoreLabel, splitCitations } from "../lib";
import PageModal from "./PageModal";

interface Props {
  text: string;
  /** True when the text is a validated answer, so citations render as chips. */
  validated: boolean;
  sources: Source[];
  /** When set, passage scores are shown using this mode's scale. */
  mode?: Mode;
  /** Maps document id to a label such as "A". */
  labels?: Record<string, string>;
}

export default function AnalysisResult({ text, validated, sources, mode, labels }: Props) {
  const baseId = useId();
  const [open, setOpen] = useState(false);
  const [focus, setFocus] = useState<string | null>(null);
  const [pageSource, setPageSource] = useState<Source | null>(null);
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

  return (
    <div>
      {validated ? (
        <p className="answer-text">
          {splitCitations(text).map((segment, i) =>
            segment.kind === "text" ? (
              <span key={i}>{segment.text}</span>
            ) : (
              <span key={i} className="chips">
                {segment.tags.map((tag) => {
                  const source = byTag.get(tag);
                  if (!source) {
                    return (
                      <span key={tag} className="chip invalid">
                        {tag}
                      </span>
                    );
                  }
                  const label = labels?.[source.document_id];
                  return (
                    <button
                      key={tag}
                      className="chip"
                      onClick={() => jumpTo(tag)}
                      title={`${source.filename}, page ${source.page_number}`}
                    >
                      {tag} · {label ? `${label} ` : ""}p.{source.page_number}
                    </button>
                  );
                })}
              </span>
            ),
          )}
        </p>
      ) : (
        <p className="muted">{text}</p>
      )}

      {sources.length > 0 && (
        <details open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
          <summary>Passages used ({sources.length})</summary>
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
                    <strong>{source.tag}</strong>
                    {labels?.[source.document_id] && ` · Document ${labels[source.document_id]}`}
                    {" · "}
                    {source.filename} · page {source.page_number}
                  </span>
                  <span className="muted small">
                    {mode && scoreLabel(mode, source.score)}
                    {source.cited && (mode ? " · cited" : "cited")}
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