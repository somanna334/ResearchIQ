import { describe, expect, it } from "vitest";
import { buildHistory } from "./history";
import type { TurnLike } from "./history";

function answered(question: string, answer: string): TurnLike {
  return { question, state: "done", response: { status: "answered", answer } };
}

describe("buildHistory", () => {
  it("returns nothing for an empty conversation", () => {
    expect(buildHistory([])).toEqual([]);
  });

  it("includes only answered turns", () => {
    const turns: TurnLike[] = [
      answered("What is a chloroplast?", "It makes sugar [S1]."),
      {
        question: "refused one",
        state: "done",
        response: { status: "insufficient_evidence", answer: "I can't answer this." },
      },
      {
        question: "withheld one",
        state: "done",
        response: { status: "citation_error", answer: "withheld" },
      },
      { question: "broken one", state: "error" },
      { question: "still loading", state: "loading" },
    ];

    expect(buildHistory(turns)).toEqual([
      { role: "user", text: "What is a chloroplast?" },
      { role: "assistant", text: "It makes sugar." },
    ]);
  });

  it("strips citation tags from answers", () => {
    const history = buildHistory([
      answered("Q?", "First [S1] and second [S1, S2] and third [ S3 ]."),
    ]);

    expect(history[1].text).toBe("First and second and third.");
    expect(history[1].text).not.toMatch(/\[S/);
  });

  it("keeps only the most recent exchanges, oldest first", () => {
    const turns = ["a", "b", "c", "d", "e"].map((x) => answered(`q-${x}`, `answer-${x}`));

    const history = buildHistory(turns, 3);

    expect(history.map((h) => h.text)).toEqual([
      "q-c", "answer-c", "q-d", "answer-d", "q-e", "answer-e",
    ]);
    expect(history.map((h) => h.role)).toEqual([
      "user", "assistant", "user", "assistant", "user", "assistant",
    ]);
  });

  it("truncates long text to the character limit", () => {
    const history = buildHistory([answered("q", "word ".repeat(400))], 3, 600);

    expect(history[1].text.length).toBeLessThanOrEqual(600);
    expect(history[1].text.endsWith("…")).toBe(true);
  });
});
