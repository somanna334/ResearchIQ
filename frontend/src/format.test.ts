import { describe, expect, it } from "vitest";
import { ciText, deltaText, money, pct, pctCi } from "./format";

describe("format helpers", () => {
  it("formats percentages and intervals", () => {
    expect(pct(0.833)).toBe("83%");
    expect(pctCi({ mean: 0.5, low: 0.25, high: 0.75 })).toBe("50% [25%, 75%]");
    expect(pctCi(null)).toBe("–");
  });

  it("formats means with intervals", () => {
    expect(ciText({ mean: 0.8123, low: 0.7, high: 0.9 })).toBe("0.81 [0.70, 0.90]");
    expect(ciText(null)).toBe("–");
  });

  it("marks only significant differences with a star", () => {
    expect(deltaText({ mean: 0.1, low: 0.02, high: 0.18, significant: true })).toBe(
      "+0.10 [+0.02, +0.18] *",
    );
    expect(deltaText({ mean: -0.05, low: -0.2, high: 0.1, significant: false })).toBe(
      "-0.05 [-0.20, +0.10]",
    );
    expect(deltaText(null)).toBe("–");
  });

  it("shows small amounts of money with enough digits", () => {
    expect(money(0.00123, "USD")).toBe("USD 0.0012");
    expect(money(1.5, "USD")).toBe("USD 1.50");
  });
});