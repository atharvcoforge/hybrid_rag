import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { CornerDownLeft } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  QUERY_STAGES,
  STAGE_LABELS,
  STAGE_SOFT_MS,
  answerParts,
  applyEvent,
  emptyStageMap,
  toMarkdown,
} from "./stream.js";

const EXAMPLES = [
  "Who signed the Carbon Reduction Plan?",
  "When was the Environmental Sustainability Policy last reviewed?",
  "What is the review date of the Water Management Policy?",
  "By when does Coforge reach net zero?",
];

const SPRING = { type: "spring", stiffness: 400, damping: 35 };

export default function App() {
  const [view, setView] = useState("read");
  const [q, setQ] = useState("");
  const [asked, setAsked] = useState("");
  const [answer, setAnswer] = useState("");
  const [hits, setHits] = useState([]);
  const [meta, setMeta] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [report, setReport] = useState(null);
  const [scoring, setScoring] = useState(false);
  const [activeCite, setActiveCite] = useState(null);
  const [hoverCite, setHoverCite] = useState(null);
  const [focus, setFocus] = useState(false);
  const [docFilter, setDocFilter] = useState("");
  const [docFilters, setDocFilters] = useState([]);
  const [health, setHealth] = useState(null);
  const [traceId, setTraceId] = useState("");
  const [stages, setStages] = useState(() => emptyStageMap());
  const [devOpen, setDevOpen] = useState(false);
  const [failedStage, setFailedStage] = useState(null);
  const [copied, setCopied] = useState(false);
  const abortRef = useRef(null);
  const inputRef = useRef(null);
  const reduce = useReducedMotion();

  useEffect(() => {
    fetch("/api/eval")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => data && setReport(data))
      .catch(() => {});
    fetch("/api/health")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => data && setHealth(data))
      .catch(() => {});
    fetch("/api/docs")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        const docs = data?.documents || [];
        setDocFilters(
          docs.map((doc) => ({
            id: doc.id,
            label:
              doc.status === "superseded"
                ? `${doc.title} (superseded)`
                : doc.title || doc.filename || doc.id,
          })),
        );
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key.toLowerCase() === "k" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    if (activeCite == null) return;
    document.getElementById(`passage-${activeCite}`)?.scrollIntoView({
      behavior: reduce ? "auto" : "smooth",
      block: "center",
    });
  }, [activeCite, reduce]);

  async function onAsk(text = q) {
    const cleaned = text.trim();
    if (!cleaned || busy) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setAsked(cleaned);
    setBusy(true);
    setAnswer("");
    setHits([]);
    setMeta(null);
    setError("");
    setActiveCite(null);
    setHoverCite(null);
    setTraceId("");
    setStages(emptyStageMap());
    setFailedStage(null);
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: cleaned, doc_id: docFilter || undefined }),
        signal: ctrl.signal,
      });
      const headerTrace = res.headers.get("x-trace-id") || res.headers.get("x-request-id");
      if (headerTrace) setTraceId(headerTrace);
      if (res.status === 429) {
        setError("Busy — retrying…");
        throw new Error("Busy — retrying");
      }
      if (!res.ok || !res.body) throw new Error(res.ok ? "empty response" : `query failed (${res.status})`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          applyEvent(part, {
            setAnswer,
            setHits,
            setMeta,
            setError,
            setTraceId,
            setStages,
          });
        }
      }
    } catch (err) {
      if (err?.name !== "AbortError") setError(err?.message || "request failed");
    } finally {
      if (abortRef.current === ctrl) setBusy(false);
    }
  }

  function onStop() {
    abortRef.current?.abort();
    setBusy(false);
  }

  async function runEval() {
    setScoring(true);
    try {
      const res = await fetch("/api/eval", { method: "POST" });
      if (!res.ok) throw new Error(`eval failed (${res.status})`);
      setReport(await res.json());
    } catch (err) {
      setError(err?.message || "eval failed");
    } finally {
      setScoring(false);
    }
  }

  const abstain = meta?.reason === "no_confident_hit";
  const verification = meta?.verification;
  const conflict = meta?.conflict || verification?.conflict;
  const lit = hoverCite ?? activeCite;
  const phase = busy && !answer ? (hits.length ? "drafting an answer from the retrieved passages" : "retrieving passages") : null;
  const degradeMsgs = [
    ...(health?.messages || []),
    ...(meta?.degraded || []),
  ].filter(Boolean);
  const statusText = error
    ? error
    : phase
      ? phase
      : verification?.state
        ? `verification: ${verification.state.replaceAll("_", " ")}`
        : abstain
          ? "declined — nothing above the confidence threshold"
          : meta?.extractive
            ? "extractive — passages only"
            : asked && answer
              ? "answer ready"
              : "";

  const runningStage = QUERY_STAGES.map((id) => stages[id]).find(
    (s) => s.state === "running" || s.state === "slow",
  );
  const liveStatus = error
    ? error
    : runningStage
      ? `${STAGE_LABELS[runningStage.id] || runningStage.id} ${runningStage.state === "slow" ? "slow" : "running"}`
      : statusText;

  async function copyTrace() {
    if (!traceId) return;
    try {
      await navigator.clipboard?.writeText(traceId);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Reading room</h1>
          <span className="tag">grounded retrieval</span>
        </div>
        <nav>
          <button className={view === "read" ? "on" : ""} onClick={() => setView("read")} type="button">
            Ask
          </button>
          <button className={view === "board" ? "on" : ""} onClick={() => setView("board")} type="button">
            Scores
          </button>
        </nav>
      </header>

      {degradeMsgs.length > 0 && (
        <div className="banner" role="status">
          {degradeMsgs[0]}
        </div>
      )}

      <p className="sr-status" role="status" aria-live="polite" aria-atomic="true">
        {liveStatus}
      </p>

      {view === "read" ? (
        <div className="desk">
          <section className="col col-ask scroll-thin">
            <div>
              <div className={`ask-shell${focus ? " focus" : ""}`}>
                <span className="ask-label">ASK</span>
                <input
                  ref={inputRef}
                  type="text"
                  value={q}
                  placeholder="What does the policy say about…"
                  spellCheck={false}
                  onFocus={() => setFocus(true)}
                  onBlur={() => setFocus(false)}
                  onChange={(event) => setQ(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") onAsk();
                  }}
                />
                {busy ? (
                  <button className="run stop" type="button" onClick={onStop}>
                    Stop
                  </button>
                ) : q.trim() ? (
                  <motion.button
                    className="run"
                    type="button"
                    onClick={() => onAsk()}
                    initial={reduce ? false : { opacity: 0, x: 4 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={SPRING}
                  >
                    Run <CornerDownLeft size={12} strokeWidth={1.75} />
                  </motion.button>
                ) : (
                  <kbd className="kbd">⌘K</kbd>
                )}
              </div>
              <StageRail
                stages={stages}
                busy={busy}
                reduce={reduce}
                onFailClick={(stage) => setFailedStage(stage)}
              />
              <div className="trace-row">
                <span className="label">Trace</span>
                <code className="trace-id num">{traceId || "—"}</code>
                <button type="button" className="copy-trace" disabled={!traceId} onClick={copyTrace}>
                  {copied ? "Copied" : "Copy"}
                </button>
                <button
                  type="button"
                  className={`dev-toggle${devOpen ? " on" : ""}`}
                  aria-pressed={devOpen}
                  onClick={() => setDevOpen((v) => !v)}
                >
                  Spans
                </button>
              </div>
              {devOpen && <DevDrawer stages={stages} traceId={traceId} />}
              {failedStage && (
                <FailedPanel
                  stage={failedStage}
                  traceId={traceId}
                  onClose={() => setFailedStage(null)}
                />
              )}
              <div className="examples">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => {
                      setQ(example);
                      inputRef.current?.focus();
                    }}
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>

            {asked && (
              <div>
                <span className="label">Question</span>
                <p className="question-echo" style={{ marginTop: 8 }}>{asked}</p>
              </div>
            )}

            <div className="panel">
              <div className="panel-head">
                <span className="label">Answer</span>
                <div className="panel-meta">
                  {verification && (
                    <span className={`chip chip-${verification.state}`}>
                      {verification.state.replaceAll("_", " ")}
                    </span>
                  )}
                  {meta && (
                    <p className="meta-line num">
                      {meta.cached ? "cached" : meta.mode}
                      {meta.cached ? "" : ` · ${meta.retrieve_ms} ms`}
                      {!meta.cached && meta.first_token_ms != null ? ` · ttft ${meta.first_token_ms} ms` : ""}
                    </p>
                  )}
                </div>
              </div>
              <div className="panel-body">
                {!asked && !busy && <p className="idle-copy">Every answer cites a passage on the right, or the system declines.</p>}
                {error && <p className="error">{error}</p>}
                {conflict && !busy && (
                  <div className="banner conflict-banner" role="status">
                    Version conflict: answering from {conflict.current_doc} ({conflict.current_value});
                    superseded {conflict.superseded_doc} states {conflict.superseded_value}.
                  </div>
                )}
                {abstain && !busy && (
                  <div className="abstain">
                    <p>The documents do not say.</p>
                    <p className="abstain-why">
                      {verification?.reason
                        ? `Stopped by the ${verification.reason.replaceAll("_", " ")} check.`
                        : "Nothing retrieved cleared the confidence threshold."}
                      {hits.length > 0 ? " Top passages are still shown so you can judge for yourself." : ""}
                    </p>
                  </div>
                )}
                <p className="answer-text">
                  {answer ? (
                    <AnswerText
                      answer={answer}
                      hitCount={hits.length}
                      lit={lit}
                      verification={verification}
                      onFocus={setActiveCite}
                      onHover={setHoverCite}
                    />
                  ) : null}
                  {busy && !reduce && (
                    <span className="caret" aria-hidden="true" />
                  )}
                  {busy && reduce && <span className="caret static" aria-hidden="true" />}
                </p>
                {answer && (
                  <button
                    className="copy"
                    type="button"
                    onClick={() => navigator.clipboard?.writeText(toMarkdown(answer, hits))}
                  >
                    Copy with citations
                  </button>
                )}
              </div>
            </div>
          </section>

          <aside className="col col-source">
            <div className="source-head">
              <span className="label">Evidence</span>
              <div className="docs" role="group" aria-label="Document filter">
                <button
                  type="button"
                  className={`doc-pill${docFilter === "" ? " on" : ""}`}
                  onClick={() => setDocFilter("")}
                >
                  All
                </button>
                {docFilters.map((doc) => (
                  <button
                    type="button"
                    className={`doc-pill${docFilter === doc.id ? " on" : ""}`}
                    key={doc.id}
                    onClick={() => setDocFilter(doc.id)}
                  >
                    {doc.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="source-body scroll-thin">
              {hits.length === 0 && (
                <p className="empty">
                  {abstain ? "No confident passages." : busy ? "Waiting for retrieved passages…" : "Ask a question. Cited passages land here."}
                </p>
              )}
              <AnimatePresence>
                {groupHits(hits).map((group) => (
                  <motion.section
                    className="doc-group"
                    key={group.key}
                    initial={reduce ? false : { opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={SPRING}
                  >
                    <h3>{group.title}</h3>
                    {group.hits.map((hit) => (
                      <motion.button
                        className={`passage${lit === hit.cite ? " lit" : ""}${hit.superseded ? " superseded" : ""}`}
                        id={`passage-${hit.cite}`}
                        key={`${hit.parent_id}-${hit.cite}`}
                        type="button"
                        onClick={() => setActiveCite(hit.cite)}
                        animate={
                          !reduce && activeCite === hit.cite
                            ? { backgroundColor: ["rgba(229,160,13,0.45)", "rgba(229,160,13,0.18)"] }
                            : undefined
                        }
                        transition={{ duration: 0.55, ease: [0.2, 0, 0, 1] }}
                      >
                        <div className="src">
                          <span className="cite-num">[{hit.cite}]</span>
                          <span>{passageMeta(hit, meta)}</span>
                          {hit.derived && <span className="chip chip-derived">derived</span>}
                          {hit.ocr && <span className="chip chip-ocr">ocr</span>}
                          {hit.superseded && (
                            <span className="chip chip-superseded">
                              superseded{hit.superseded_by ? ` by ${hit.superseded_by}` : ""}
                            </span>
                          )}
                        </div>
                        <p>{hit.text}</p>
                      </motion.button>
                    ))}
                  </motion.section>
                ))}
              </AnimatePresence>
            </div>
          </aside>
        </div>
      ) : (
        <Board report={report} scoring={scoring} onRun={runEval} />
      )}
    </div>
  );
}

function AnswerText({ answer, hitCount, lit, verification, onFocus, onHover }) {
  const unsupported = new Set(verification?.unsupported || []);
  return answerParts(answer, hitCount).map((part) =>
    part.type === "cite" ? (
      <button
        className={`cite${lit === part.n ? " lit" : ""}${verification?.state === "verified" ? " ok" : ""}`}
        data-cite={part.n}
        key={part.key}
        type="button"
        aria-label={`Passage ${part.n}`}
        onClick={() => onFocus(part.n)}
        onMouseEnter={() => onHover(part.n)}
        onMouseLeave={() => onHover(null)}
      >
        {part.n}
      </button>
    ) : (
      <span
        key={part.key}
        className={unsupported.has(part.text.trim()) ? "unsupported" : undefined}
        title={unsupported.has(part.text.trim()) ? "not supported by the cited passage" : undefined}
      >
        {part.text}
      </span>
    ),
  );
}

function StageRail({ stages, busy, reduce, onFailClick }) {
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!busy || reduce) return undefined;
    const id = window.setInterval(() => setTick((n) => n + 1), 100);
    return () => window.clearInterval(id);
  }, [busy, reduce]);

  if (!busy && QUERY_STAGES.every((id) => stages[id].state === "pending")) return null;

  return (
    <ol className="stage-rail" aria-label="Query pipeline stages">
      {QUERY_STAGES.map((id) => {
        const stage = stages[id];
        const soft = STAGE_SOFT_MS[id] ?? 5000;
        let state = stage.state;
        let elapsed = stage.duration_ms;
        if ((state === "running" || state === "slow") && stage.startedAt) {
          elapsed = Date.now() - stage.startedAt;
          if (elapsed >= soft) state = "slow";
        }
        const label = STAGE_LABELS[id] || id;
        const title =
          stage.cache_hit || stage.reason === "cached"
            ? "cached"
            : state === "skipped"
              ? stage.reason?.startsWith("skipped")
                ? stage.reason
                : `skipped — ${stage.reason || "unavailable"}`
              : state === "failed"
                ? stage.error || "failed"
                : label;
        const display =
          stage.cache_hit || stage.reason === "cached"
            ? "cached"
            : state === "skipped"
              ? stage.reason?.includes("unavailable")
                ? "skipped — unavailable"
                : stage.reason?.includes("not in mode")
                  ? "skipped — not in mode"
                  : "skipped"
              : null;
        const clickable = state === "failed";
        const Tag = clickable ? "button" : "span";
        return (
          <li key={id} className={`stage stage-${state}`} data-stage={id}>
            <Tag
              type={clickable ? "button" : undefined}
              className="stage-btn"
              title={title}
              aria-label={`${label}: ${state}${elapsed != null ? `, ${Math.round(elapsed)} milliseconds` : ""}`}
              onClick={clickable ? () => onFailClick(stage) : undefined}
            >
              <span className="stage-name">{label}</span>
              {display ? (
                <span className="stage-note">{display}</span>
              ) : elapsed != null ? (
                <span className="stage-ms num">{Math.round(elapsed)}ms</span>
              ) : state === "pending" ? (
                <span className="stage-ms num">—</span>
              ) : null}
            </Tag>
          </li>
        );
      })}
    </ol>
  );
}

function DevDrawer({ stages, traceId }) {
  return (
    <div className="dev-drawer" role="region" aria-label="Span table">
      <div className="dev-head">
        <span className="label">Span table</span>
        <code className="trace-id num">{traceId || "—"}</code>
      </div>
      <table>
        <thead>
          <tr>
            <th>stage</th>
            <th>state</th>
            <th>ms</th>
            <th>in</th>
            <th>out</th>
            <th>top</th>
            <th>cache</th>
          </tr>
        </thead>
        <tbody>
          {QUERY_STAGES.map((id) => {
            const s = stages[id];
            return (
              <tr key={id} className={`stage-${s.state}`}>
                <td className="num">{id}</td>
                <td>{s.state}</td>
                <td className="num">{s.duration_ms != null ? Math.round(s.duration_ms) : "—"}</td>
                <td className="num">{s.candidates_in ?? "—"}</td>
                <td className="num">{s.candidates_out ?? "—"}</td>
                <td className="num">{s.top_score != null ? Number(s.top_score).toFixed(4) : "—"}</td>
                <td className="num">{s.cache_hit == null ? "—" : s.cache_hit ? "hit" : "miss"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function FailedPanel({ stage, traceId, onClose }) {
  return (
    <div className="fail-panel" role="dialog" aria-label="Stage failure">
      <div className="fail-head">
        <strong>{STAGE_LABELS[stage.id] || stage.id} failed</strong>
        <button type="button" onClick={onClose} aria-label="Close">
          Close
        </button>
      </div>
      <p className="num">{stage.error_type || "Error"}</p>
      <p>{stage.error || "unknown error"}</p>
      <p className="meta-line">
        trace <code className="trace-id num">{traceId || "—"}</code>
      </p>
    </div>
  );
}

function groupHits(hits) {
  const groups = [];
  const by = new Map();
  hits.forEach((hit, index) => {
    const cite = hit.cite || index + 1;
    const title = hit.title || hit.source || "Document";
    if (!by.has(title)) {
      const group = { key: title, title, hits: [] };
      by.set(title, group);
      groups.push(group);
    }
    by.get(title).hits.push({ ...hit, cite });
  });
  return groups;
}

function passageMeta(hit, meta) {
  const bits = [];
  if (hit.heading) bits.push(hit.heading);
  if (hit.page_start) bits.push(`p.${hit.page_start}`);
  if (Number.isFinite(hit.score)) bits.push(hit.score.toFixed(3));
  if (hit.confident) bits.push("agreed");
  else if (meta?.mode === "cascade") bits.push("uncertain");
  return bits.join(" · ");
}

function Board({ report, scoring, onRun }) {
  const rows = report?.scores || [];
  const overall = rows.filter((row) => row.kind === "all");
  return (
    <div className="board-wrap scroll-thin">
      <button type="button" onClick={onRun} disabled={scoring}>
        {scoring ? "Scoring" : "Run eval"}
      </button>
      {report?.verdict && <p className="verdict">{report.verdict}</p>}
      {overall.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>mode</th>
              <th>kind</th>
              <th>recall@5</th>
              <th>MRR</th>
              <th>abstain</th>
              <th>p50</th>
              <th>p95</th>
              <th>p99</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.mode}-${row.kind}`}>
                <td className="num">{row.mode}{report.live_mode === row.mode && row.kind === "all" ? " · live" : ""}</td>
                <td className="num">{row.kind}</td>
                <td className="num">{(row.recall * 100).toFixed(0)}%</td>
                <td className="num">{row.mrr.toFixed(2)}</td>
                <td className="num">{(row.abstain * 100).toFixed(0)}%</td>
                <td className="num">{Math.round(row.p50)} ms</td>
                <td className="num">{Math.round(row.p95)} ms</td>
                <td className="num">{Math.round(row.p99)} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {report?.span_hit != null && (
        <p className="meta-line num" style={{ margin: "16px 0 0" }}>
          answer span-hit {(report.span_hit * 100).toFixed(0)}% on {report.span_n} rows
        </p>
      )}
    </div>
  );
}
