"""Live view of the robot's VLM in a browser: the live camera feed, and every question the model was asked
about a frame with its answer (open questions in words, closed ones with their probabilities).

The robot streams frames to POST /feed (shown as they arrive; no model runs on them) and asks about the
newest frame with POST /ask (a whole question set per frame). Older single-question endpoints
(/is_visible, /where, /is_near) are shown too. A page at http://<host>:<port>/view polls for news.
Closed answers are probabilities scored from the next token (see smolvlm.py); the page shows the answer
the model leans to (p >= 0.5). The executor applies its own calibrated thresholds.

    GET /view                    the page
    GET /view/events?after=<id>  JSON: queries newer than <id> (no images) and the feed status
    GET /view/image/<id>.png     the camera image of query <id>
    GET /view/feed.png           the newest frame of the live feed
    POST /view/clear             forget all recorded queries (the live feed keeps running)
"""
import io
import json
import time
from collections import deque
from threading import Lock

from smolvlm import CROPS, NEAR_QUESTION, QUESTION

PROMPTS = {
    "/is_visible": lambda phrase: QUESTION.format(phrase=phrase),
    "/is_near": lambda phrase: NEAR_QUESTION.format(phrase=phrase),
    "/where": lambda phrase: QUESTION.format(phrase=phrase) + "  (asked separately on the left, center and right crops)",
    "/ask": lambda phrase: f"question set about {phrase or 'the frame'}",
}


def summarize(endpoint, reply):
    """The answer the model leans to, as text."""
    if "answers" in reply:
        answers = reply["answers"]
        parts = [f"{qid} {a['answer']}" for qid, a in answers.items() if qid in ("action", "visible")]
        return ", ".join(parts) or f"{len(answers)} answers"
    if "p_yes" in reply:
        p = reply["p_yes"]
        word = "yes" if p >= 0.5 else "no"
        return f"{word}  (p_yes = {p:.3f})"
    scores = reply["scores"]
    best = max(scores, key=scores.get)
    if scores[best] < 0.5:
        return "not in view  (" + ", ".join(f"{d} {s:.2f}" for d, s in scores.items()) + ")"
    return f"{best}  (" + ", ".join(f"{d} {s:.2f}" for d, s in scores.items()) + ")"


def question_rows(reply):
    """/ask: one row per question, in the order asked."""
    rows = []
    for q in reply.get("questions", []):
        a = reply["answers"].get(q["id"], {})
        rows.append({"id": q["id"], "kind": q["kind"], "text": q["text"], "crop": q.get("crop"),
                     "after": q.get("after"), "answer": a.get("answer"), "p_yes": a.get("p_yes"),
                     "scores": a.get("scores")})
    return rows


class LiveLog:
    def __init__(self, keep=60):
        self._entries = deque(maxlen=keep)
        self._images = {}
        self._next_id = 1
        self._lock = Lock()
        self._feed_png, self._feed = None, {}
        self._arrivals = deque(maxlen=20)  # (wall time, robot time) of recent feed frames
        self.cleared_at = 0.0  # wall time of the last clear(), so every open page resets

    def record(self, endpoint, phrase, image, reply):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        with self._lock:
            entry_id = self._next_id
            self._next_id += 1
            if len(self._entries) == self._entries.maxlen:
                self._images.pop(self._entries[0]["id"], None)
            self._entries.append({
                "id": entry_id, "time": time.time(), "endpoint": endpoint.lstrip("/"), "phrase": phrase,
                "prompt": reply.get("prompt") or PROMPTS[endpoint](phrase), "answer": summarize(endpoint, reply),
                "p_yes": reply.get("p_yes"), "scores": reply.get("scores"),
                "questions": question_rows(reply),
                "latency_s": reply.get("latency_s"), "onboard_latency_s": reply.get("onboard_latency_s"),
                "width": image.width, "height": image.height,
            })
            self._images[entry_id] = buffer.getvalue()

    def feed(self, png, frame_id, t):
        with self._lock:
            self._feed_png = png
            self._arrivals.append((time.time(), t))
            self._feed = {"frame_id": frame_id, "t": t}

    def feed_frame(self):
        with self._lock:
            return self._feed_png

    def feed_status(self):
        """Newest frame and the feed rate over the last frames, in wall-clock and robot time."""
        with self._lock:
            status = dict(self._feed)
            arrivals = list(self._arrivals)
        if not status:
            return None
        status["age_s"] = round(time.time() - arrivals[-1][0], 2)
        if len(arrivals) >= 2:
            (w0, t0), (w1, t1) = arrivals[0], arrivals[-1]
            n = len(arrivals) - 1
            status["wall_hz"] = round(n / (w1 - w0), 2) if w1 > w0 else None
            status["robot_hz"] = round(n / (t1 - t0), 2) if t1 > t0 else None
        return status

    def clear(self):
        """Forget every recorded query and its image. Ids keep counting up, so open pages stay in sync."""
        with self._lock:
            self._entries.clear()
            self._images.clear()
            self.cleared_at = time.time()
            return self._next_id - 1

    def events_after(self, after):
        with self._lock:
            return [e for e in self._entries if e["id"] > after]

    def image(self, entry_id):
        with self._lock:
            return self._images.get(entry_id)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VLM Live View</title>
<style>
:root { --bg:#f6f7f9; --panel:#fff; --text:#1c2128; --muted:#5f6b7a; --line:#dde2e8; --yes:#1f8a4c; --no:#b4372d; --accent:#2f6fd0; }
@media (prefers-color-scheme: dark) { :root { --bg:#121417; --panel:#1b1f24; --text:#e6e9ee; --muted:#9aa5b1; --line:#2c333b; --yes:#4cc27e; --no:#ef6f63; --accent:#6ea2f0; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui, sans-serif; }
header { padding:12px 16px; border-bottom:1px solid var(--line); display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; }
header h1 { font-size:17px; margin:0; } header span { color:var(--muted); font-size:13px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--muted); margin-right:6px; }
.dot.live { background:var(--yes); }
main { display:grid; grid-template-columns:minmax(0, 1fr) minmax(0, 1.2fr) minmax(220px, .8fr); gap:16px; padding:16px; }
@media (max-width: 1200px) { main { grid-template-columns:minmax(0, 1fr) minmax(0, 1fr); } aside { grid-column:1 / -1; } }
@media (max-width: 760px) { main { grid-template-columns:1fr; } }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; min-width:0; }
h2 { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:0 0 8px; font-weight:600; }
.frame { position:relative; width:100%; aspect-ratio:4/3; background:#000; border-radius:6px; overflow:hidden; }
.frame img { width:100%; height:100%; object-fit:contain; display:block; }
.crop { position:absolute; top:0; bottom:0; border:2px solid var(--accent); opacity:.85; }
.crop b { position:absolute; bottom:4px; left:4px; font-size:12px; background:var(--accent); color:#fff; padding:1px 5px; border-radius:3px; }
.label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; margin:12px 0 4px; }
.meta { color:var(--muted); font-size:13px; margin-top:8px; }
.prompt { white-space:pre-wrap; font-family:ui-monospace, monospace; font-size:13px; background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:8px 10px; }
.answer { font-size:20px; font-weight:600; }
.yes { color:var(--yes); } .no { color:var(--no); }
.qa { display:flex; flex-direction:column; gap:10px; }
.q { border-top:1px solid var(--line); padding-top:8px; }
.q:first-child { border-top:0; padding-top:0; }
.q .text { font-size:13px; color:var(--muted); white-space:pre-wrap; }
.q .text summary { cursor:pointer; }
.q .words { font-size:16px; font-style:italic; margin-top:2px; }
.q .closed { font-size:16px; font-weight:600; margin-top:2px; }
.tag { font-size:11px; border:1px solid var(--line); border-radius:4px; padding:0 4px; margin-right:6px; color:var(--muted); font-style:normal; font-weight:400; }
.bars { display:grid; gap:4px; margin-top:6px; }
.bar { display:grid; grid-template-columns:70px 1fr 48px; gap:8px; align-items:center; font-size:12px; }
.track { height:8px; background:var(--bg); border:1px solid var(--line); border-radius:4px; overflow:hidden; }
.fill { height:100%; background:var(--accent); }
.history { display:flex; flex-direction:column; gap:8px; max-height:80vh; overflow-y:auto; }
.item { display:grid; grid-template-columns:80px 1fr; gap:10px; padding:6px; border-radius:8px; border:1px solid transparent; cursor:pointer; }
.item:hover, .item.sel { border-color:var(--line); background:var(--bg); }
.item img { width:80px; aspect-ratio:4/3; object-fit:cover; border-radius:4px; background:#000; }
.item .s { font-size:13px; overflow-wrap:anywhere; }
header button { margin-left:auto; font:inherit; font-size:13px; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:4px 12px; cursor:pointer; }
header button:hover { border-color:var(--accent); }
header button:disabled { opacity:.6; cursor:default; }
.empty { color:#9aa5b1; padding:24px 12px; text-align:center; position:absolute; inset:0; display:flex; align-items:center; justify-content:center; }
</style></head>
<body>
<header><h1>VLM Live View</h1><span id="model">connecting…</span>
<span><span id="dot" class="dot"></span><span id="status">waiting for the robot</span></span>
<label style="font-size:13px;color:var(--muted)"><input type="checkbox" id="follow" checked> follow latest</label>
<button id="clear" type="button">Clear history</button></header>
<main>
  <section class="panel">
    <h2>Live camera feed</h2>
    <div class="frame"><img id="feed" alt="live camera frame" hidden><div class="empty" id="feedEmpty">No frames yet. Run a mission with --approach live.</div></div>
    <div class="meta" id="feedMeta">–</div>
    <h2 style="margin-top:16px">Frame the model answered about</h2>
    <div class="frame" id="frame"><div class="empty" id="placeholder">No questions yet.</div></div>
    <div class="meta" id="meta"></div>
  </section>
  <section class="panel">
    <h2>Questions and answers</h2>
    <div id="answers"><div class="meta">–</div></div>
  </section>
  <aside class="panel"><h2>Recent analyses</h2><div class="history" id="history"></div></aside>
</main>
<script>
const CROPS = __CROPS__;
let last = 0, selected = null, lastSeen = 0, feedId = -1, clearedAt = 0;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const bar = (k, v) => `<div class="bar"><span>${esc(k)}</span><div class="track"><div class="fill" style="width:${(v*100).toFixed(1)}%"></div></div><span>${v.toFixed(3)}</span></div>`;
function cls(e) {
  if (e.questions && e.questions.length) { const v = e.questions.find(q => q.id === "visible"); return v ? v.answer : "yes"; }
  return e.answer.startsWith("yes") || ["left","center","right"].some(d => e.answer.startsWith(d)) ? "yes" : "no";
}
function qaHtml(e) {
  if (e.questions && e.questions.length) return `<div class="qa">` + e.questions.map(q => {
    const where = q.crop ? ` <span class="tag">crop ${Math.round(q.crop[0]*100)}–${Math.round(q.crop[1]*100)}%</span>` : "";
    const after = q.after ? ` <span class="tag">after "${esc(q.after)}"</span>` : "";
    const text = q.text.length > 120 ? `<details><summary>${esc(q.text.slice(0, 100))}…</summary>${esc(q.text)}</details>` : esc(q.text);
    let body;
    if (q.kind === "open") body = `<div class="words">“${esc(q.answer)}”</div>`;
    else if (q.kind === "yesno") body = `<div class="closed ${q.answer}">${q.answer}</div><div class="bars">${bar("p(yes)", q.p_yes)}</div>`;
    else body = `<div class="closed">${esc(q.answer)}</div><div class="bars">${Object.entries(q.scores).map(([k,v]) => bar(k, v)).join("")}</div>`;
    return `<div class="q"><div class="text"><span class="tag">${q.id} · ${q.kind}</span>${where}${after}${text}</div>${body}</div>`;
  }).join("") + `</div>`;
  const vals = e.scores ? e.scores : {"p(yes)": e.p_yes};
  return `<div class="label">Question</div><div class="prompt">${esc(e.prompt)}</div>` +
    `<div class="label">Answer</div><div class="answer ${cls(e)}">${esc(e.answer.split("  ")[0])}</div><div class="bars">${Object.entries(vals).map(([k,v]) => bar(k, v)).join("")}</div>`;
}
function show(e) {
  selected = e.id;
  const f = $("frame"); f.innerHTML = `<img src="/view/image/${e.id}.png" alt="analysed frame">`;
  if (e.endpoint === "where") for (const [d,[x0,x1]] of Object.entries(CROPS))
    f.insertAdjacentHTML("beforeend", `<div class="crop" style="left:${x0*100}%;width:${(x1-x0)*100}%"><b>${d} ${e.scores[d].toFixed(2)}</b></div>`);
  $("answers").innerHTML = qaHtml(e);
  const onboard = e.onboard_latency_s && Math.abs(e.onboard_latency_s - e.latency_s) > 1e-3 ? ` · charged onboard ${e.onboard_latency_s.toFixed(2)} s` : "";
  const n = e.questions && e.questions.length ? ` · ${e.questions.length} questions` : "";
  $("meta").textContent = `#${e.id} · ${e.endpoint} · ${e.phrase || "–"}${n} · ${e.width}×${e.height}px · model ${(e.latency_s*1000).toFixed(0)} ms${onboard} · ${new Date(e.time*1000).toLocaleTimeString()}`;
  document.querySelectorAll(".item").forEach(x => x.classList.toggle("sel", +x.dataset.id === e.id));
}
function addHistory(e) {
  const div = document.createElement("div"); div.className = "item"; div.dataset.id = e.id;
  const scene = (e.questions || []).find(q => q.id === "scene");
  div.innerHTML = `<img src="/view/image/${e.id}.png" alt=""><div><div class="s">${esc(e.endpoint)}: ${esc(e.phrase)}</div><div class="s" style="font-weight:600;color:var(--${cls(e)})">${esc(e.answer.split("  ")[0])}</div>` +
    (scene ? `<div class="s" style="font-style:italic">${esc(scene.answer)}</div>` : "") +
    `<div class="s" style="color:var(--muted)">#${e.id} · ${new Date(e.time*1000).toLocaleTimeString()}</div></div>`;
  div.onclick = () => { $("follow").checked = false; show(e); };
  $("history").prepend(div);
  while ($("history").children.length > 60) $("history").lastChild.remove();
}
function resetAnswers() {
  selected = null;
  $("history").innerHTML = "";
  $("answers").innerHTML = `<div class="meta">–</div>`;
  $("meta").textContent = "";
  $("frame").innerHTML = `<div class="empty" id="placeholder">No questions yet.</div>`;
}
$("clear").onclick = async () => {
  if (!confirm("Clear the history of analysed frames for everyone viewing this page?")) return;
  $("clear").disabled = true;
  try {
    const r = await fetch("/view/clear", {method: "POST"});
    if (!r.ok) throw new Error(r.status);
    last = Math.max(last, (await r.json()).last_id);
    resetAnswers();
  } catch { alert("Could not clear the history: server not reachable."); }
  $("clear").disabled = false;
};
function showFeed(s) {
  if (!s) return;
  if (s.frame_id !== feedId) { feedId = s.frame_id; $("feed").src = `/view/feed.png?f=${s.frame_id}`; $("feed").hidden = false; $("feedEmpty").hidden = true; }
  const rate = [s.robot_hz != null ? `${s.robot_hz} frames/s robot time` : null, s.wall_hz != null ? `${s.wall_hz} frames/s wall clock` : null].filter(Boolean).join(" · ");
  $("feedMeta").textContent = `frame #${s.frame_id} · robot t = ${s.t.toFixed(2)} s` + (rate ? ` · ${rate}` : "") + (s.age_s > 3 ? ` · last frame ${s.age_s.toFixed(0)} s ago` : "");
}
async function poll() {
  try {
    const r = await fetch(`/view/events?after=${last}`); const data = await r.json();
    $("model").textContent = `${data.model} · profile ${data.profile}`;
    showFeed(data.feed);
    if (data.cleared_at > clearedAt) { clearedAt = data.cleared_at; resetAnswers(); }
    for (const e of data.events) { addHistory(e); last = e.id; lastSeen = Date.now(); }
    if (data.events.length) { $("placeholder")?.remove(); if ($("follow").checked || selected === null) show(data.events[data.events.length-1]); }
    const live = Date.now() - lastSeen < 5000 || (data.feed && data.feed.age_s < 5);
    $("dot").className = "dot" + (live ? " live" : ""); $("status").textContent = live ? "live" : (lastSeen || data.feed ? "idle" : "waiting for the robot");
  } catch { $("status").textContent = "server not reachable"; $("dot").className = "dot"; }
  setTimeout(poll, 200);
}
poll();
</script></body></html>
"""


def page():
    return PAGE.replace("__CROPS__", json.dumps(CROPS))
