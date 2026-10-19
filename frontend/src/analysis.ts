import { request } from "./api";
import type { AnswerStatus, Mode, Source } from "./api";

export interface Usage {
  model: string;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  retrieval_ms: number;
  llm_ms: number;
}

export interface SummaryResponse {
  status: AnswerStatus;
  summary: string;
  sources: Source[];
  coverage: {
    chunks_used: number;
    chunks_total: number;
    pages_covered: number[];
    page_count: number;
    sampled: boolean;
  };
  usage: Usage;
}

export interface CompareDocument {
  document_id: string;
  filename: string;
  label: string;
  passages: number;
  best_dense_score: number | null;
  cited: boolean;
}

export interface CompareResponse {
  status: AnswerStatus;
  answer: string;
  documents: CompareDocument[];
  sources: Source[];
  usage: Usage;
  retrieval: { mode: Mode; min_top_score: number; gated: boolean };
}

export interface CompareRequest {
  document_ids: string[];
  question: string;
  k: number;
  mode: Mode;
}

export function summarizeDocument(id: string): Promise<SummaryResponse> {
  return request<SummaryResponse>(`/documents/${id}/summary`, { method: "POST" });
}

export function compareDocuments(body: CompareRequest): Promise<CompareResponse> {
  return request<CompareResponse>("/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}