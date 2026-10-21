import { request } from "./api";

export type SplitName = "test" | "dev" | "all";

export interface Ci {
  mean: number;
  low: number;
  high: number;
  n: number;
}

export interface Delta {
  mean: number;
  low: number;
  high: number;
  significant: boolean;
}

export interface RunItem {
  id: string;
  kind: "retrieval" | "answers";
  created_utc: string;
  git_commit: string | null;
  questions: number;
  label: string;
}

export interface RunMeta {
  run_id: string;
  created_utc: string;
  git_commit: string | null;
  git_dirty?: boolean;
  k: number;
  mode?: string;
  chunk_config?: string;
  dataset: { path: string; sha256: string; questions: number };
  corpus: { filename: string; pages: number }[];
  generation?: { model: string };
  judge?: { enabled: boolean; model: string | null };
}

export interface RetrievalRow {
  name: string;
  n: number;
  metrics: Record<string, Ci | null>;
  delta: { recall: Delta | null; mrr: Delta | null } | null;
  latency: { p50: number; p95: number } | null;
}

export interface GateData {
  answerable: number;
  negatives: number;
  current: number;
  rows: { threshold: number; false_refuse: number; false_accept: number }[];
}

export interface RetrievalDetail {
  id: string;
  kind: "retrieval";
  meta: RunMeta;
  k: number;
  splits: Record<SplitName, RetrievalRow[]>;
  gate: Record<SplitName, GateData | null>;
}

export interface AnswerSummary {
  n_answerable: number;
  n_unanswerable: number;
  errors: number;
  judge_failures: number;
  answerable: { answered: Ci | null; refused: Ci | null; citation_error: Ci | null };
  unanswerable: { refused: Ci | null; answered: Ci | null; citation_error: Ci | null };
  correctness: {
    judged: number;
    verdicts: { correct: number; partial: number; incorrect: number };
    correct_of_judged: Ci | null;
    mean_score: Ci | null;
    end_to_end_correct: Ci | null;
  };
  faithfulness: {
    n: number;
    per_answer: Ci | null;
    claim_level: { supported: number; total: number };
  };
  citation: { precision: Ci | null; recall: Ci | null; hits_relevant: Ci | null };
  latency: { p50: number; p95: number } | null;
  tokens: {
    questions: number;
    generation_prompt: number;
    generation_completion: number;
    judge_prompt: number;
    judge_completion: number;
  };
  cost: { currency: string; total: number; per_question: number | null } | null;
  mean_total_tokens: number | null;
}

export interface AnswersDetail {
  id: string;
  kind: "answers";
  meta: RunMeta;
  splits: Record<SplitName, AnswerSummary>;
}

export type RunDetail = RetrievalDetail | AnswersDetail;

export function listRuns(): Promise<RunItem[]> {
  return request<RunItem[]>("/eval/runs");
}

export function getRun(id: string): Promise<RunDetail> {
  return request<RunDetail>(`/eval/runs/${id}`);
}