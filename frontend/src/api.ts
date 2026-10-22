import { clearToken, getToken, notifyUnauthorized } from "./session";

export const API_URL: string =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type DocStatus = "processing" | "ready" | "no_text" | "failed";
export type Mode = "semantic" | "hybrid" | "hybrid_rerank";
export type AnswerStatus = "answered" | "insufficient_evidence" | "citation_error";

export interface DocumentInfo {
  id: string;
  filename: string;
  size_bytes: number;
  sha256: string;
  status: DocStatus;
  error_message: string | null;
  page_count: number;
  pages_with_text: number;
  created_at: string;
}

export interface EmbedStats {
  chunk_config: string;
  model_name: string;
  total_chunks: number;
  newly_embedded: number;
  seconds: number;
}

export interface PageInfo {
  page_number: number;
  text: string;
  char_count: number;
}

export interface Source {
  tag: string;
  chunk_id: number;
  document_id: string;
  filename: string;
  page_number: number;
  score: number; // meaning depends on the retrieval mode
  text: string;
  cited: boolean;
}

export interface AskResponse {
  status: AnswerStatus;
  answer: string;
  sources: Source[];
  usage: {
    model: string;
    prompt_tokens: number | null;
    completion_tokens: number | null;
    retrieval_ms: number;
    llm_ms: number;
  };
  retrieval: {
    mode: Mode;
    best_dense_score: number | null;
    min_top_score: number;
    gated: boolean;
  };
}

export interface AskRequest {
  question: string;
  k: number;
  mode: Mode;
  document_ids?: string[];
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "Cannot reach the backend. Is it running on " + API_URL + "?");
  }

  if (res.status === 401 && token) {
    // The session expired or was ended elsewhere: return to the sign-in screen.
    clearToken();
    notifyUnauthorized();
  }

  if (!res.ok) {
    let detail = `Request failed (HTTP ${res.status})`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        detail = body.detail
          .map((d: { msg?: string }) => d.msg ?? "invalid input")
          .join("; ");
      }
    } catch {
      // The error body wasn't JSON; keep the generic message.
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export function listDocuments(): Promise<DocumentInfo[]> {
  return request<DocumentInfo[]>("/documents");
}

export function uploadDocument(file: File): Promise<DocumentInfo> {
  const form = new FormData();
  form.append("file", file); // the browser sets the multipart Content-Type itself
  return request<DocumentInfo>("/documents", { method: "POST", body: form });
}

export function deleteDocument(id: string): Promise<void> {
  return request<void>(`/documents/${id}`, { method: "DELETE" });
}

export function indexDocument(id: string): Promise<EmbedStats> {
  return request<EmbedStats>(`/documents/${id}/embeddings`, { method: "POST" });
}

export function getPage(documentId: string, pageNumber: number): Promise<PageInfo> {
  return request<PageInfo>(`/documents/${documentId}/pages/${pageNumber}`);
}

export function ask(body: AskRequest): Promise<AskResponse> {
  return request<AskResponse>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}