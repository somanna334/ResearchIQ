import { describe, expect, it } from "vitest";
import { toggleLimited } from "./selection";

describe("toggleLimited", () => {
  it("adds an id that is not selected", () => {
    expect(toggleLimited(["a"], "b", 3)).toEqual(["a", "b"]);
  });

  it("removes an id that is already selected", () => {
    expect(toggleLimited(["a", "b"], "a", 3)).toEqual(["b"]);
  });

  it("ignores additions beyond the limit", () => {
    expect(toggleLimited(["a", "b", "c"], "d", 3)).toEqual(["a", "b", "c"]);
  });
});