/** Pure stream/citation helpers — kept free of React so tests stay cheap. */

export const QUERY_STAGES = [
  "embed_query",
  "dense",
  "lexical",
  "fuse",
  "rerank",
  "gate",
  "generate_first_token",
  "generate_complete",
  "gate_form",
  "gate_literal",
  "gate_entail",
  "gate_coverage",
  "gate_conflict",
];

export const STAGE_LABELS = {
  embed_query: "embed",
  dense: "dense",
  lexical: "lexical",
  fuse: "fuse",
  rerank: "rerank",
  gate: "gate",
  generate_first_token: "first token",
  generate_complete: "generate",
  gate_form: "form",
  gate_literal: "literal",
  gate_entail: "entail",
  gate_coverage: "coverage",
  gate_conflict: "conflict",
};

/** Soft budgets (ms) — keep in sync with rag.telemetry.DEFAULT_SOFT_MS. */
export const STAGE_SOFT_MS = {
  embed_query: 400,
  dense: 100,
  lexical: 100,
  fuse: 50,
  rerank: 1000,
  gate: 100,
  generate_first_token: 1200,
  generate_complete: 10000,
  gate_form: 50,
  gate_literal: 100,
  gate_entail: 500,
  gate_coverage: 50,
  gate_conflict: 50,
};

export function emptyStageMap() {
  return Object.fromEntries(
    QUERY_STAGES.map((id) => [
      id,
      {
        id,
        state: "pending",
        duration_ms: null,
        startedAt: null,
        candidates_in: null,
        candidates_out: null,
        top_score: null,
        cache_hit: null,
        model: null,
        skipped: false,
        reason: "",
        error: null,
        error_type: null,
      },
    ]),
  );
}

export function reduceStage(prev, payload) {
  const id = payload?.stage;
  if (!id || !(id in prev)) return prev;
  const cur = { ...prev[id] };
  const event = payload.event;
  if (event === "start") {
    cur.state = "running";
    cur.startedAt = Date.now();
    cur.duration_ms = null;
    cur.error = null;
    cur.error_type = null;
    if (payload.skipped) {
      cur.skipped = true;
      cur.reason = payload.reason || cur.reason;
    }
  } else if (event === "stage_slow") {
    if (cur.state === "running" || cur.state === "slow") cur.state = "slow";
    if (payload.duration_ms != null) cur.duration_ms = payload.duration_ms;
  } else if (event === "finish") {
    if (payload.skipped) {
      cur.state = "skipped";
      cur.skipped = true;
      cur.reason = payload.reason || "skipped";
    } else {
      cur.state = "done";
    }
    cur.duration_ms = payload.duration_ms ?? cur.duration_ms;
    cur.startedAt = null;
  } else if (event === "error") {
    cur.state = "failed";
    cur.duration_ms = payload.duration_ms ?? cur.duration_ms;
    cur.error = payload.error || payload.message || "failed";
    cur.error_type = payload.error_type || null;
    cur.startedAt = null;
  }
  for (const key of ["candidates_in", "candidates_out", "top_score", "cache_hit", "model", "degraded"]) {
    if (payload[key] != null) cur[key] = payload[key];
  }
  if (payload.reason && event !== "start") cur.reason = payload.reason;
  return { ...prev, [id]: cur };
}

/** Mark stages that never started once the request is finished. */
export function finalizeSkipped(stages, mode) {
  const next = { ...stages };
  for (const id of QUERY_STAGES) {
    const cur = next[id];
    if (cur.state !== "pending") continue;
    let reason = "not reached";
    if (id === "rerank" && mode && mode !== "cascade" && mode !== "rerank") {
      reason = "skipped — not in mode";
    }
    next[id] = { ...cur, state: "skipped", skipped: true, reason };
  }
  return next;
}

export function markCached(stages) {
  const next = { ...stages };
  for (const id of QUERY_STAGES) {
    next[id] = {
      ...next[id],
      state: "done",
      duration_ms: next[id].duration_ms ?? 0,
      cache_hit: true,
      skipped: false,
      reason: "cached",
    };
  }
  return next;
}

export function answerParts(answer, hitCount) {
  const parts = [];
  const re = /\[(\d+)\]/g;
  let last = 0;
  let key = 0;
  for (const match of answer.matchAll(re)) {
    if (match.index > last) parts.push({ type: "text", text: answer.slice(last, match.index), key: key++ });
    const n = Number(match[1]);
    if (n >= 1 && n <= hitCount) parts.push({ type: "cite", n, key: key++ });
    else parts.push({ type: "text", text: match[0], key: key++ });
    last = match.index + match[0].length;
  }
  if (last < answer.length) parts.push({ type: "text", text: answer.slice(last), key: key++ });
  return parts;
}

export function toMarkdown(answer, hits) {
  const lines = [answer.trim(), "", "Sources:"];
  hits.forEach((hit) => {
    const pages = hit.page_start ? ` p.${hit.page_start}` : "";
    lines.push(`- [${hit.cite}] ${hit.source || hit.title}${pages}`);
  });
  return lines.join("\n");
}

export function applyEvent(part, a, b, c, d) {
  const handlers =
    typeof a === "function" || a == null
      ? { setAnswer: a, setHits: b, setMeta: c, setError: d }
      : a;
  const { setAnswer, setHits: setHitsFn, setMeta: setMetaFn, setError: setErrorFn, setTraceId, setStages } = handlers;

  let event = "message";
  let data = "";
  for (const line of part.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;
  let payload;
  try {
    payload = JSON.parse(data);
  } catch {
    setErrorFn?.("bad stream chunk");
    return;
  }
  if (event === "trace") {
    setTraceId?.(payload.trace_id || payload.request_id || "");
  } else if (event === "stage") {
    setStages?.((prev) => reduceStage(prev || emptyStageMap(), payload));
  } else if (event === "meta") {
    setHitsFn?.(payload.hits || []);
    setMetaFn?.(payload);
    if (payload.trace_id || payload.request_id) {
      setTraceId?.(payload.trace_id || payload.request_id);
    }
    if (payload.extractive) {
      setStages?.((prev) => {
        let next = prev || emptyStageMap();
        for (const id of ["generate_first_token", "generate_complete", "gate_form", "gate_literal", "gate_entail", "gate_coverage", "gate_conflict"]) {
          if (next[id]?.state === "pending") {
            next = reduceStage(next, {
              stage: id,
              event: "finish",
              skipped: true,
              reason: "skipped — unavailable",
            });
          }
        }
        return next;
      });
    }
  } else if (event === "token") {
    setAnswer?.((prev) => prev + payload.t);
  } else if (event === "done") {
    setMetaFn?.((prev) => ({
      ...(prev || {}),
      first_token_ms: payload.first_token_ms,
      cached: prev?.cached || payload.cached,
      verification: payload.verification || prev?.verification,
      conflict: payload.conflict || prev?.conflict || payload.verification?.conflict,
      extractive: payload.extractive || prev?.extractive,
      trace_id: payload.trace_id || prev?.trace_id,
      mode: payload.mode || prev?.mode,
    }));
    if (payload.answer) setAnswer?.(payload.answer);
    if (payload.cached) {
      setStages?.((prev) => markCached(prev || emptyStageMap()));
    } else {
      setStages?.((prev) => finalizeSkipped(prev || emptyStageMap(), payload.mode));
    }
  } else if (event === "error") {
    setErrorFn?.(payload.message || "writer error");
    setMetaFn?.((prev) => ({
      ...(prev || {}),
      error: payload.message,
      trace_id: payload.trace_id || prev?.trace_id,
    }));
    if (payload.stage) {
      setStages?.((prev) =>
        reduceStage(prev || emptyStageMap(), {
          stage: payload.stage,
          event: "error",
          error: payload.message,
          error_type: payload.detail,
        }),
      );
    }
    setStages?.((prev) => finalizeSkipped(prev || emptyStageMap(), null));
  }
}
