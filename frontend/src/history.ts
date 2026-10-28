import type { HistoryTurn } from "./api";

/** The part of a chat turn that buildHistory needs (the Chat component's turns fit it). */
export interface TurnLike {
  question: string;
  state: "loading" | "done" | "error";
  response?: { status: string; answer: string };
}

// Same pattern as the backend validator: [S1], [S1, S2]. Global, so every one goes.
const CITATION_TAGS = /\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]/g;

function clean(text: string, maxChars: number): string {
  const stripped = text
    .replace(CITATION_TAGS, "")
    .replace(/\s+/g, " ")
    .replace(/\s+([.,;:!?])/g, "$1") // "sugar [S1]." would otherwise leave "sugar ."
    .trim();
  return stripped.length > maxChars ? stripped.slice(0, maxChars - 1).trimEnd() + "…" : stripped;
}

/**
 * The earlier conversation, in the form the server wants, for rewriting follow-ups.
 *
 * Only turns that were ANSWERED count. Refusals, citation errors, request errors and
 * turns still loading say nothing a follow-up could refer to, and a refusal would
 * only teach the rewriter to ask the question the same way again.
 */
export function buildHistory(
  turns: TurnLike[],
  maxExchanges = 3,
  maxChars = 600,
): HistoryTurn[] {
  const answered = turns.filter(
    (t) => t.state === "done" && t.response?.status === "answered",
  );
  if (maxExchanges <= 0) return [];
  const history: HistoryTurn[] = [];
  for (const turn of answered.slice(-maxExchanges)) {
    const question = clean(turn.question, maxChars);
    const answer = clean(turn.response?.answer ?? "", maxChars);
    if (!question || !answer) continue;
    history.push({ role: "user", text: question });
    history.push({ role: "assistant", text: answer });
  }
  return history;
}
