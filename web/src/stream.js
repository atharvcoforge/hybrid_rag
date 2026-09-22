/** Pure stream/citation helpers — kept free of React so tests stay cheap. */

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

export function applyEvent(part, setAnswer, setHits, setMeta, setError) {
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
    setError("bad stream chunk");
    return;
  }
  if (event === "meta") {
    setHits(payload.hits || []);
    setMeta(payload);
  } else if (event === "token") {
    setAnswer((prev) => prev + payload.t);
  } else if (event === "done") {
    setMeta((prev) => ({
      ...(prev || {}),
      first_token_ms: payload.first_token_ms,
      cached: prev?.cached || payload.cached,
      verification: payload.verification || prev?.verification,
      extractive: payload.extractive || prev?.extractive,
    }));
    if (payload.answer) setAnswer(payload.answer);
  } else if (event === "error") {
    setError(payload.message || "writer error");
    setMeta((prev) => ({ ...(prev || {}), error: payload.message }));
  }
}
