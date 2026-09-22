import { describe, expect, it, vi } from "vitest";
import { answerParts, applyEvent, toMarkdown } from "./stream.js";

describe("answerParts", () => {
  it("turns in-range citations into cite parts", () => {
    const parts = answerParts("Signed by Alice [1].", 2);
    expect(parts.some((part) => part.type === "cite" && part.n === 1)).toBe(true);
  });

  it("leaves out-of-range citations as text", () => {
    const parts = answerParts("Bad cite [9].", 2);
    expect(parts.every((part) => part.type === "text")).toBe(true);
  });
});

describe("applyEvent", () => {
  it("ignores a trailing empty buffer chunk", () => {
    const setError = vi.fn();
    applyEvent("", vi.fn(), vi.fn(), vi.fn(), setError);
    expect(setError).not.toHaveBeenCalled();
  });

  it("surfaces a bad JSON chunk", () => {
    const setError = vi.fn();
    applyEvent("event: meta\ndata: {bad", vi.fn(), vi.fn(), vi.fn(), setError);
    expect(setError).toHaveBeenCalledWith("bad stream chunk");
  });
});

describe("toMarkdown", () => {
  it("keeps sources with the answer", () => {
    const md = toMarkdown("Net zero by 2040 [1].", [
      { cite: 1, source: "Carbon_New_2040.pdf", page_start: 3 },
    ]);
    expect(md).toContain("Net zero by 2040 [1].");
    expect(md).toContain("Carbon_New_2040.pdf p.3");
  });
});
