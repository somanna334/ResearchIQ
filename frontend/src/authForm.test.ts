import { describe, expect, it } from "vitest";
import { normalizeEmail, validateRegistration } from "./authForm";

describe("validateRegistration", () => {
  it("accepts a valid email and a long matching password", () => {
    expect(validateRegistration("ann@example.com", "correct-horse-battery", "correct-horse-battery")).toBeNull();
  });

  it("rejects malformed emails", () => {
    expect(validateRegistration("ann", "correct-horse-battery", "correct-horse-battery")).toMatch(/email/i);
    expect(validateRegistration("a b@example.com", "correct-horse-battery", "correct-horse-battery")).toMatch(/email/i);
  });

  it("rejects short passwords", () => {
    expect(validateRegistration("ann@example.com", "short", "short")).toMatch(/at least 10/);
  });

  it("rejects mismatched confirmation", () => {
    expect(validateRegistration("ann@example.com", "correct-horse-battery", "correct-horse-batterz")).toMatch(/match/);
  });
});

describe("normalizeEmail", () => {
  it("trims and lowercases", () => {
    expect(normalizeEmail("  Ann@Example.COM ")).toBe("ann@example.com");
  });
});