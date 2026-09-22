import { useEffect, useState } from "react";
import { motion } from "framer-motion";

const DOCS = ["Carbon Reduction Plan", "Environmental Sustainability Policy", "Water Management Policy"];

export default function App() {
  const [view, setView] = useState("read");
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [hits, setHits] = useState([]);
  const [meta, setMeta] = useState(null);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState(null);
  const [scoring, setScoring] = useState(false);

  useEffect(() => {
    fetch("/api/eval")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => data && setReport(data))
      .catch(() => {});
  }, []);

  async function onAsk(event) {
    event.preventDefault();
    const text = q.trim();
    if (!text || busy) return;
    setBusy(true);
    setAnswer("");
    setHits([]);
    setMeta(null);
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q: text }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) applyEvent(part, setAnswer, setHits, setMeta);
    }
    setBusy(false);
  }

  async function runEval() {
    setScoring(true);
    const res = await fetch("/api/eval", { method: "POST" });
    setReport(await res.json());
    setScoring(false);
  }

  return (
    <div className="app">
      <main className="sheet">
        <header>
          <h1>Reading room</h1>
          <div className="docs">
            {DOCS.map((name) => (
              <div key={name}>{name}</div>
            ))}
          </div>
        </header>
        <nav>
          <button className={view === "read" ? "on" : ""} onClick={() => setView("read")} type="button">
            Ask
          </button>
          <button className={view === "board" ? "on" : ""} onClick={() => setView("board")} type="button">
            Scores
          </button>
        </nav>
        {view === "read" ? (
          <>
            <form onSubmit={onAsk}>
              <input
                type="text"
                value={q}
                placeholder="Ask the three policies"
                onChange={(event) => setQ(event.target.value)}
              />
              <button type="submit" disabled={busy}>
                {busy ? "Reading" : "Ask"}
              </button>
            </form>
            <div className="answer">
              {answer || (meta?.reason === "no_confident_hit" ? "The documents do not say." : "")}
            </div>
            {meta && (
              <div className="meta">
                {meta.mode} · retrieve {meta.retrieve_ms} ms
                {meta.first_token_ms != null ? ` · first token ${meta.first_token_ms} ms` : ""}
                {meta.error ? ` · ${meta.error}` : ""}
              </div>
            )}
          </>
        ) : (
          <Board report={report} scoring={scoring} onRun={runEval} />
        )}
      </main>
      <aside className="rail">
        <h2>Passages</h2>
        {hits.length === 0 && <div className="empty">Nothing retrieved yet.</div>}
        {hits.map((hit, index) => (
          <motion.article
            className="card"
            key={hit.parent_id + index}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.04 }}
          >
            <div className="src">
              [{index + 1}] {hit.source}
              {hit.page_start ? ` · p.${hit.page_start}` : ""} · {hit.score.toFixed(3)}
              {hit.confident ? " · agreed" : meta?.mode === "cascade" ? <span className="flag"> · uncertain</span> : ""}
            </div>
            <p>{hit.text}</p>
          </motion.article>
        ))}
      </aside>
    </div>
  );
}

function applyEvent(part, setAnswer, setHits, setMeta) {
  let event = "message";
  let data = "";
  for (const line of part.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;
  const payload = JSON.parse(data);
  if (event === "meta") {
    setHits(payload.hits || []);
    setMeta(payload);
  } else if (event === "token") {
    setAnswer((prev) => prev + payload.t);
  } else if (event === "done") {
    setMeta((prev) => ({ ...(prev || {}), first_token_ms: payload.first_token_ms }));
  } else if (event === "error") {
    setMeta((prev) => ({ ...(prev || {}), error: payload.message }));
  }
}

function Board({ report, scoring, onRun }) {
  const rows = (report?.scores || []).filter((row) => row.kind === "all");
  return (
    <div className="board">
      <button type="button" onClick={onRun} disabled={scoring}>
        {scoring ? "Scoring" : "Run eval"}
      </button>
      {report?.verdict && <p className="verdict">{report.verdict}</p>}
      {rows.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>mode</th>
              <th>recall@5</th>
              <th>MRR</th>
              <th>abstain</th>
              <th>p50</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.mode}>
                <td>{row.mode}{report.live_mode === row.mode ? " · live" : ""}</td>
                <td>
                  {(row.recall * 100).toFixed(0)}%
                  <motion.div
                    className="bar"
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: row.recall }}
                  />
                </td>
                <td>{row.mrr.toFixed(2)}</td>
                <td>{(row.abstain * 100).toFixed(0)}%</td>
                <td>{Math.round(row.p50)} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {report?.span_hit != null && (
        <p className="meta">answer span-hit {(report.span_hit * 100).toFixed(0)}% on {report.span_n} rows</p>
      )}
    </div>
  );
}
