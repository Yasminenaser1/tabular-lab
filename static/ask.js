// The Ask AI page. Moved out of the wizard's floating panel when the site
// became multi-page; the request/response contract with POST /ask is unchanged.
(function () {
  const log = document.getElementById("chat-log");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  const offline = document.getElementById("offline");
  const history = [];

  // The server caps a request at 20 messages; keep the tail well inside that.
  const MAX_TURNS = 16;

  function bubble(text, who, { tools, kind } = {}) {
    const el = document.createElement("div");
    el.className = `bubble bubble-${who}` + (kind ? ` bubble-${kind}` : "");
    el.textContent = text;
    if (tools && tools.length) {
      const row = document.createElement("div");
      row.className = "tools";
      for (const t of tools) {
        const tag = document.createElement("span");
        tag.className = "tool-badge";
        tag.textContent = t;
        row.appendChild(tag);
      }
      el.appendChild(row);
    }
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }

  function looksLikeRawToolCall(text) {
    // The small model sometimes prints a tool call as text instead of calling it.
    // Catch that so the user never sees a raw JSON blob.
    const t = (text || "").trim();
    if (!t) return true;
    if (t.startsWith("{") && /"(name|parameters|fields|changes)"\s*:/.test(t)) return true;
    return false;
  }

  async function send(question) {
    const q = question.trim();
    if (!q) return;
    input.value = "";
    bubble(q, "you");
    history.push({ role: "user", content: q });
    if (history.length > MAX_TURNS) history.splice(0, history.length - MAX_TURNS);

    sendBtn.disabled = true;
    const thinking = bubble("Thinking…", "ai", { kind: "wait" });
    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history }),
      });
      const data = await res.json();
      thinking.remove();
      if (!res.ok) {
        bubble(`Error ${res.status}: ${data.detail || "something went wrong"}`, "ai", { kind: "error" });
      } else if (looksLikeRawToolCall(data.answer)) {
        bubble("Sorry, I couldn't put that into words. Try rephrasing your question.", "ai");
      } else {
        bubble(data.answer, "ai", { tools: data.tools_used });
        history.push({ role: "assistant", content: data.answer });
      }
    } catch {
      thinking.remove();
      bubble("Couldn't reach the assistant.", "ai", { kind: "error" });
    }
    sendBtn.disabled = false;
    input.focus();
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    send(input.value);
  });

  for (const b of document.querySelectorAll("[data-suggestion]")) {
    b.addEventListener("click", () => send(b.textContent));
  }

  // If no backend is configured, say so rather than letting every send fail.
  fetch("/ask/status")
    .then(r => r.json())
    .then(s => { if (!s.available) disable("The assistant isn't configured on this deployment."); })
    .catch(() => disable("Couldn't reach the assistant service."));

  function disable(message) {
    offline.textContent = message + " The rest of the site works normally — try the prediction wizard.";
    offline.hidden = false;
    input.disabled = true;
    sendBtn.disabled = true;
    for (const b of document.querySelectorAll("[data-suggestion]")) b.disabled = true;
  }

  bubble("Ask me about this model — how accurate it is, what drives a prediction, or the risk for a patient you describe.", "ai");
  input.focus();
})();
