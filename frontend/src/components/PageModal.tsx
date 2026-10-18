import { useEffect, useRef, useState } from "react";
import { getPage } from "../api";
import type { PageInfo, Source } from "../api";
import { splitHighlight } from "../lib";

interface Props {
  source: Source;
  onClose: () => void;
}

export default function PageModal({ source, onClose }: Props) {
  const [page, setPage] = useState<PageInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const markRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let cancelled = false;
    getPage(source.document_id, source.page_number)
      .then((p) => {
        if (!cancelled) setPage(p);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Could not load the page");
      });
    return () => {
      cancelled = true;
    };
  }, [source.document_id, source.page_number]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    if (page) markRef.current?.scrollIntoView({ block: "center" });
  }, [page]);

  const parts = page ? splitHighlight(page.text, source.text) : null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label={`${source.filename}, page ${source.page_number}`}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="modal-head">
          <strong>
            {source.filename} · page {source.page_number}
          </strong>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>
        <div className="modal-body">
          {error && <div className="banner bad">{error}</div>}
          {!page && !error && <p className="muted">Loading page…</p>}
          {page && parts && (
            <p className="page-text">
              {parts.before}
              <mark ref={markRef}>{parts.match}</mark>
              {parts.after}
            </p>
          )}
          {page && !parts && (
            <>
              <p className="muted small">
                The retrieved passage couldn't be located on this page, so nothing is highlighted.
              </p>
              <p className="page-text raw">{page.text}</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}