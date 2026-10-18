import { useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import { deleteDocument, indexDocument, uploadDocument } from "../api";
import type { DocumentInfo } from "../api";
import { formatBytes, sha256Hex } from "../lib";

// Mirrors the backend default. This is a convenience; the backend enforces the real limit.
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

interface Props {
  documents: DocumentInfo[];
  selected: string[];
  onToggle: (id: string) => void;
  onChanged: () => Promise<void>;
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong";
}

export default function DocumentPanel({ documents, selected, onToggle, onChanged }: Props) {
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [workingId, setWorkingId] = useState<string | null>(null);

  async function handleFiles(files: File[]) {
    if (files.length === 0) return;
    setError(null);
    setBusy(true);
    try {
      for (const file of files) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
          setError(`${file.name}: only PDF files are accepted`);
          continue;
        }
        if (file.size > MAX_UPLOAD_BYTES) {
          setError(`${file.name}: larger than ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB`);
          continue;
        }
        const hash = await sha256Hex(file);
        if (hash && documents.some((d) => d.sha256 === hash && d.status === "ready")) {
          setError(`${file.name}: this exact file is already uploaded`);
          continue;
        }

        setNote(`Uploading ${file.name}…`);
        const doc = await uploadDocument(file);
        if (doc.status === "ready") {
          setNote(`Indexing ${file.name}… (the first time can take a minute while the model loads)`);
          await indexDocument(doc.id);
        } else {
          setError(`${file.name}: ${doc.error_message ?? doc.status}`);
        }
      }
    } catch (err) {
      setError(messageOf(err));
    } finally {
      setBusy(false);
      setNote(null);
      await onChanged();
    }
  }

  function onPick(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []); // copy before resetting the input
    event.target.value = "";
    void handleFiles(files);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragOver(false);
    void handleFiles(Array.from(event.dataTransfer.files));
  }

  async function reindex(doc: DocumentInfo) {
    setWorkingId(doc.id);
    setError(null);
    setNote(`Indexing ${doc.filename}…`);
    try {
      const stats = await indexDocument(doc.id);
      setNote(`${doc.filename}: ${stats.newly_embedded} new vectors (${stats.total_chunks} chunks)`);
    } catch (err) {
      setNote(null);
      setError(messageOf(err));
    } finally {
      setWorkingId(null);
    }
  }

  async function remove(doc: DocumentInfo) {
    if (!window.confirm(`Delete "${doc.filename}" and its index? This cannot be undone.`)) return;
    setWorkingId(doc.id);
    setError(null);
    try {
      await deleteDocument(doc.id);
      await onChanged();
    } catch (err) {
      setError(messageOf(err));
    } finally {
      setWorkingId(null);
    }
  }

  return (
    <div className="docs">
      <div
        className={"dropzone" + (dragOver ? " over" : "")}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <p>Drop PDFs here</p>
        <label className={"primary file-btn" + (busy ? " disabled" : "")}>
          Choose files
          <input type="file" accept="application/pdf,.pdf" multiple hidden disabled={busy} onChange={onPick} />
        </label>
      </div>

      {note && <p className="muted small">{note}</p>}
      {error && <div className="banner bad small">{error}</div>}

      <h2 className="section-title">
        Documents ({documents.length})
        {selected.length > 0 && <span className="muted small"> · {selected.length} selected</span>}
      </h2>
      {documents.length === 0 && <p className="muted small">No documents yet.</p>}
      <ul className="doc-list">
        {documents.map((doc) => (
          <li key={doc.id} className="doc">
            <label className="doc-main">
              <input
                type="checkbox"
                checked={selected.includes(doc.id)}
                disabled={doc.status !== "ready"}
                onChange={() => onToggle(doc.id)}
              />
              <span className="doc-name" title={doc.filename}>
                {doc.filename}
              </span>
            </label>
            <div className="doc-meta muted small">
              <span className={"status " + doc.status} title={doc.error_message ?? undefined}>
                {doc.status === "no_text" ? "no text" : doc.status}
              </span>
              <span>
                {doc.page_count} pages · {formatBytes(doc.size_bytes)}
              </span>
            </div>
            {doc.error_message && doc.status !== "ready" && (
              <p className="small muted">{doc.error_message}</p>
            )}
            <div className="doc-actions">
              {doc.status === "ready" && (
                <button className="link-btn" disabled={workingId === doc.id || busy} onClick={() => void reindex(doc)}>
                  Re-index
                </button>
              )}
              <button className="link-btn danger" disabled={workingId === doc.id || busy} onClick={() => void remove(doc)}>
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>
      <p className="muted small">
        Nothing selected means all documents are searched. Re-index is safe to repeat: it only
        computes missing vectors.
      </p>
    </div>
  );
}