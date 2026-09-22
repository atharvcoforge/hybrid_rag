import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { CornerDownLeft } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { answerParts, applyEvent, toMarkdown } from "./stream.js";

const DOC_FILTERS = [
  { id: "Carbon_New_2040.pdf", label: "Carbon Reduction Plan" },
  { id: "Environmental_Sustainability_Policy_2025.pdf", label: "Environmental Sustainability Policy" },
  { id: "Water-Management-Policy.pdf", label: "Water Management Policy" },
];
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
  const [health, setHealth] = useState(null);
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
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: cleaned, doc_id: docFilter || undefined }),
        signal: ctrl.signal,
      });
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
        for (const part of parts) applyEvent(part, setAnswer, setHits, setMeta, setError);
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

      <p className="sr-status" aria-live="polite">{statusText}</p>

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
              <StageRail meta={meta} busy={busy} />
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
                  {busy && (
                    <motion.span
                      className="caret"
                      aria-hidden="true"
                      animate={reduce ? {} : { opacity: [1, 0.15, 1] }}
                      transition={{ duration: 0.9, repeat: Infinity, ease: "linear" }}
                    />
                  )}
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
                {DOC_FILTERS.map((doc) => (
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
                        className={`passage${lit === hit.cite ? " lit" : ""}`}
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

function StageRail({ meta, busy }) {
  const stages = meta?.stages_ms || {};
  const steps = [
    { id: "retrieve", label: "retrieve", ms: meta?.retrieve_ms },
    { id: "rerank", label: "rerank", ms: stages.rerank },
    { id: "gate", label: "gate", ms: stages.gate },
    { id: "ttft", label: "first token", ms: meta?.first_token_ms },
  ];
  if (!busy && !meta) return null;
  return (
    <ol className="stage-rail" aria-label="Pipeline stages">
      {steps.map((step) => (
        <li key={step.id} className={step.ms != null ? "done" : busy ? "wait" : ""}>
          <span>{step.label}</span>
          {step.ms != null && <span className="num">{step.ms} ms</span>}
        </li>
      ))}
    </ol>
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
