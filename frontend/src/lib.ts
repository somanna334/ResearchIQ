import type { Mode } from "./api";

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "citations"; tags: string[] };

// Same pattern as the backend validator: [S1], [S1, S2]; adjacent [S1][S2] match separately.
const CITATION = /\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]/;

/** Split an answer into plain text and citation groups. */
export function splitCitations(answer: string): Segment[] {
  // split() with one capture group alternates: text, captured tags, text, ...
  const parts = answer.split(CITATION);
  const segments: Segment[] = [];
  parts.forEach((part, index) => {
    if (index % 2 === 0) {
      if (part) segments.push({ kind: "text", text: part });
    } else {
      segments.push({ kind: "citations", tags: part.split(",").map((t) => t.trim()) });
    }
  });
  return segments;
}

/** Scores have different meanings per mode, so label them. */
export function scoreLabel(mode: Mode, score: number): string {
  if (mode === "semantic") return `cosine ${score.toFixed(3)}`;
  if (mode === "hybrid") return `RRF ${score.toFixed(4)}`;
  return `rerank ${score.toFixed(2)}`;
}

export function collapseWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

export interface HighlightParts {
  before: string;
  match: string;
  after: string;
}

/**
 * Locate a retrieved passage inside its page. Chunks are cut from
 * whitespace-collapsed page text, so we compare in that form.
 */
export function splitHighlight(pageText: string, passage: string): HighlightParts | null {
  const page = collapseWhitespace(pageText);
  const needle = collapseWhitespace(passage);
  if (!needle) return null;
  const start = page.indexOf(needle);
  if (start === -1) return null;
  return {
    before: page.slice(0, start),
    match: needle,
    after: page.slice(start + needle.length),
  };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** SHA-256 of a file, to spot duplicate uploads. Returns null if the browser can't. */
export async function sha256Hex(file: File): Promise<string | null> {
  try {
    const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  } catch {
    return null;
  }
}