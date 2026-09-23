import { describe, expect, it, vi } from "vitest";
import {
  answerParts,
  applyEvent,
  emptyStageMap,
  finalizeSkipped,
  reduceStage,
  toMarkdown,
} from "./stream.js";

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

  it("applies trace and stage events", () => {
    const setTraceId = vi.fn();
    const setStages = vi.fn((fn) => fn(emptyStageMap()));
    applyEvent('event: trace\ndata: {"trace_id":"abc"}', { setTraceId, setStages });
    expect(setTraceId).toHaveBeenCalledWith("abc");
    applyEvent(
      'event: stage\ndata: {"stage":"dense","event":"start"}',
      { setTraceId, setStages },
    );
    expect(setStages).toHaveBeenCalled();
  });
});

describe("reduceStage", () => {
  it("moves start → slow → finish", () => {
    let map = emptyStageMap();
    map = reduceStage(map, { stage: "dense", event: "start" });
    expect(map.dense.state).toBe("running");
    map = reduceStage(map, { stage: "dense", event: "stage_slow", duration_ms: 150 });
    expect(map.dense.state).toBe("slow");
    map = reduceStage(map, { stage: "dense", event: "finish", duration_ms: 200, candidates_out: 5 });
    expect(map.dense.state).toBe("done");
    expect(map.dense.candidates_out).toBe(5);
  });

  it("records skipped finish honestly", () => {
    let map = emptyStageMap();
    map = reduceStage(map, {
      stage: "rerank",
      event: "finish",
      skipped: true,
      reason: "skipped — unavailable",
    });
    expect(map.rerank.state).toBe("skipped");
    expect(map.rerank.reason).toContain("unavailable");
  });
});

describe("finalizeSkipped", () => {
  it("marks pending rerank as not-in-mode for rrf", () => {
    const map = finalizeSkipped(emptyStageMap(), "rrf");
    expect(map.rerank.state).toBe("skipped");
    expect(map.rerank.reason).toContain("not in mode");
  });
});

describe("toMarkdown", () => {
  it("keeps sources with the answer", () => {
    const md = toMarkdown("Net zero by 2040 [1].", [
      { cite: 1, source: "Carbon_Reduction_Plan.pdf", page_start: 3 },
    ]);
    expect(md).toContain("Net zero by 2040 [1].");
    expect(md).toContain("Carbon_Reduction_Plan.pdf p.3");
  });
});
