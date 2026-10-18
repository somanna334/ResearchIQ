import { describe, expect, it } from "vitest";
import { scoreLabel, splitCitations, splitHighlight } from "./lib";

describe("splitCitations", () => {
  it("splits plain, grouped and adjacent tags", () => {
    expect(splitCitations("A [S1] B [S2, S3][S4].")).toEqual([
      { kind: "text", text: "A " },
      { kind: "citations", tags: ["S1"] },
      { kind: "text", text: " B " },
      { kind: "citations", tags: ["S2", "S3"] },
      { kind: "citations", tags: ["S4"] },
      { kind: "text", text: "." },
    ]);
  });

  it("leaves text without tags untouched", () => {
    expect(splitCitations("No citations here.")).toEqual([
      { kind: "text", text: "No citations here." },
    ]);
  });

  it("ignores brackets that are not citation tags", () => {
    const text = "See [source] and [S] and [S1";
    expect(splitCitations(text)).toEqual([{ kind: "text", text }]);
  });
});

describe("splitHighlight", () => {
  it("ignores whitespace differences between page and passage", () => {
    expect(splitHighlight("Hello\nworld  of   text", "world of")).toEqual({
      before: "Hello ",
      match: "world of",
      after: " text",
    });
  });

  it("returns null when the passage is not on the page", () => {
    expect(splitHighlight("Hello world", "goodbye")).toBeNull();
  });

  it("returns null for an empty passage", () => {
    expect(splitHighlight("Hello world", "   ")).toBeNull();
  });
});

describe("scoreLabel", () => {
  it("labels each mode's score with its own scale", () => {
    expect(scoreLabel("semantic", 0.6543)).toBe("cosine 0.654");
    expect(scoreLabel("hybrid", 0.032)).toBe("RRF 0.0320");
    expect(scoreLabel("hybrid_rerank", -6.4753)).toBe("rerank -6.48");
  });
});