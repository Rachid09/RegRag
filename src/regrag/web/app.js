// RegRAG chat page: sends a question to POST /ask and renders the streamed events
// (sources, token, done, error). All server text is inserted with textContent, never as HTML.

const $ = (sel) => document.querySelector(sel);
const thread = $("#thread");
const form = $("#composer");
const input = $("#question");
const sendBtn = $("#send");
const SEND_ICON = sendBtn.innerHTML;
const STOP_ICON =
  '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor"/></svg>';

let controller = null; // AbortController of the request in flight
let msgCount = 0; // gives each answer's source cards unique ids for [n] links

// ---- health badge ---------------------------------------------------------
async function checkHealth() {
  const el = $("#status");
  try {
    const res = await fetch("/health");
    const h = await res.json();
    el.className = "status ok";
    el.querySelector(".status-text").textContent = `Ready · ${h.model.replace(/^.*\//, "")}`;
  } catch {
    el.className = "status down";
    el.querySelector(".status-text").textContent = "Offline";
  }
}

// ---- rendering ------------------------------------------------------------
function addUser(text) {
  const node = $("#tpl-user").content.cloneNode(true);
  node.querySelector(".bubble").textContent = text;
  thread.append(node);
}

function addBot() {
  thread.append($("#tpl-bot").content.cloneNode(true));
  const msg = thread.lastElementChild;
  msg.dataset.id = String(++msgCount);
  return msg;
}

function renderSources(msg, sources) {
  const list = msg.querySelector(".source-list");
  for (const s of sources) {
    const li = document.createElement("li");
    li.className = "source";
    li.id = `src-${msg.dataset.id}-${s.rank}`;

    const head = document.createElement("div");
    head.className = "source-head";
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = s.rank;
    const cite = document.createElement("span");
    cite.className = "cite";
    cite.textContent = s.doc_title + " ";
    const pid = document.createElement("small");
    pid.textContent = `· passage ${s.passage_id}`;
    cite.append(pid);
    head.append(num, cite);
    li.append(head);

    if (s.headings.length) {
      const path = document.createElement("p");
      path.className = "path";
      path.textContent = s.headings.join(" › ");
      li.append(path);
    }

    const excerpt = document.createElement("p");
    excerpt.className = "excerpt";
    excerpt.textContent = s.text;
    li.append(excerpt);

    const more = document.createElement("button");
    more.type = "button";
    more.className = "more";
    more.textContent = "Show full passage";
    more.addEventListener("click", () => {
      const open = li.classList.toggle("open");
      more.textContent = open ? "Show less" : "Show full passage";
    });
    li.append(more);
    list.append(li);
  }
  msg.querySelector(".sources").hidden = sources.length === 0;
}

// After the answer is complete: turn "[2]" into a link to source card 2, highlight the
// cards the answer cites and dim the ones it does not.
function linkCitations(msg, answer, citations) {
  const cited = new Set(citations.map((c) => c.n));
  const text = answer.textContent;
  answer.textContent = "";
  for (const part of text.split(/(\[\d+\])/)) {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m || !cited.has(Number(m[1]))) { answer.append(part); continue; }
    const a = document.createElement("a");
    a.className = "cite-mark";
    a.href = `#src-${msg.dataset.id}-${m[1]}`;
    a.textContent = m[1];
    a.setAttribute("aria-label", `Source ${m[1]}`);
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const card = document.getElementById(a.hash.slice(1));
      card.classList.add("open", "flash");
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      setTimeout(() => card.classList.remove("flash"), 1200);
    });
    answer.append(a);
  }
  msg.querySelectorAll(".source").forEach((li, i) => {
    li.classList.add(cited.has(i + 1) ? "cited" : "uncited");
  });
  if (cited.size) msg.querySelector(".sources h3").textContent = `Sources · ${cited.size} cited`;
}

// Pin the new question under the top bar; the answer then fills in below it.
function pinToTop(el) {
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---- streaming ------------------------------------------------------------
// A server-sent event is "event: <name>\ndata: <json>\n\n"; this yields {event, data}.
async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      yield { event, data: data ? JSON.parse(data) : null };
    }
  }
}

async function ask(question) {
  $("#empty")?.remove();
  addUser(question);
  const msg = addBot();
  const answer = msg.querySelector(".answer");
  let started = false;
  setBusy(true);
  pinToTop(msg.previousElementSibling);

  controller = new AbortController();
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    if (!res.ok) {
      if (res.status === 422) throw new Error("Please ask a longer question.");
      if (res.status === 429) throw new Error((await res.json()).detail);
      throw new Error(`Server error (${res.status}).`);
    }

    for await (const { event, data } of readEvents(res)) {
      if (event === "sources") renderSources(msg, data);
      else if (event === "token") {
        if (!started) { answer.textContent = ""; started = true; }
        answer.textContent += data;
      } else if (event === "done") {
        if (data.refused) answer.classList.add("refused");
        else if (data.fallback) answer.classList.add("notice");
        else linkCitations(msg, answer, data.citations);
      } else if (event === "error") throw new Error(data.message);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      if (!started) answer.textContent = "Stopped.";
    } else {
      answer.textContent = err.message || "Something went wrong.";
      answer.classList.add("error");
    }
  } finally {
    controller = null;
    setBusy(false);
  }
}

// ---- composer -------------------------------------------------------------
function setBusy(busy) {
  sendBtn.classList.toggle("stop", busy);
  sendBtn.innerHTML = busy ? STOP_ICON : SEND_ICON;
  sendBtn.setAttribute("aria-label", busy ? "Stop" : "Send");
  updateSend();
  if (!busy) input.focus();
}

function updateSend() {
  sendBtn.disabled = !controller && input.value.trim().length < 3;
}

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (controller) { controller.abort(); return; }
  const q = input.value.trim();
  if (q.length < 3) return;
  input.value = "";
  autosize();
  ask(q);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    form.requestSubmit();
  }
});
input.addEventListener("input", () => { autosize(); updateSend(); });

document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => { if (!controller) ask(chip.textContent.trim()); })
);

updateSend();
checkHealth();
input.focus();
