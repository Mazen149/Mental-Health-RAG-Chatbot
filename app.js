const chatArea = document.getElementById("chat-area");
const input = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const settingsBtn = document.getElementById("settings-btn");
const settingsPanel = document.getElementById("settings-panel");
const settingsSave = document.getElementById("settings-save");
const settingsCancel = document.getElementById("settings-cancel");
const clearBtn = document.getElementById("clear-btn");
const apiUrlInput = document.getElementById("api-url");
const apiEndpointInput = document.getElementById("api-endpoint");

const STORAGE_KEY = "sanad_ai_settings";
const defaults = {
  apiUrl: window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "" // Relative URL for internal Docker routing
    : "https://sanad-ai.myvnc.com",
  endpoint: "/chat"
};

// ── CORS proxy for mixed-content (HTTPS page → HTTP API) ──
function proxyUrl(url) {
  const needsProxy =
    window.location.protocol === "https:" && url.startsWith("http://");
  return needsProxy
    ? "https://corsproxy.io/?" + url
    : url;
}

function loadSettings() {
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(STORAGE_KEY)) };
  } catch {
    return { ...defaults };
  }
}

function saveSettings(s) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
}

let settings = loadSettings();

// ── Settings panel ──

settingsBtn.addEventListener("click", () => {
  const open = settingsPanel.classList.toggle("open");
  if (open) {
    apiUrlInput.value = settings.apiUrl;
    apiEndpointInput.value = settings.endpoint;
    apiUrlInput.focus();
  }
});

settingsCancel.addEventListener("click", () =>
  settingsPanel.classList.remove("open")
);

settingsSave.addEventListener("click", () => {
  settings.apiUrl = apiUrlInput.value.replace(/\/+$/, "") || defaults.apiUrl;
  settings.endpoint = apiEndpointInput.value || defaults.endpoint;
  saveSettings(settings);
  settingsPanel.classList.remove("open");
});

// ── Input handling ──

input.addEventListener("input", () => {
  sendBtn.disabled = !input.value.trim();
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 120) + "px";
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (input.value.trim()) send();
  }
});

sendBtn.addEventListener("click", send);

// ── Quick prompts ──

document.addEventListener("click", (e) => {
  if (e.target.classList.contains("prompt-chip")) {
    input.value = e.target.dataset.prompt;
    input.dispatchEvent(new Event("input"));
    send();
  }
});

// ── Clear chat ──

clearBtn.addEventListener("click", () => {
  chatArea.innerHTML = `
    <div class="welcome">
      <div class="welcome-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 21C12 21 4 15 4 9.5C4 6.46 6.46 4 9.5 4C11.06 4 12.45 4.74 12 5.5C11.55 4.74 12.94 4 14.5 4C17.54 4 20 6.46 20 9.5C20 15 12 21 12 21Z"/>
        </svg>
      </div>
      <h2>Welcome to Sanad AI</h2>
      <p>A safe space to talk about how you're feeling. I'm here to listen and help with questions about anxiety, depression, stress, and more.</p>
      <div class="quick-prompts">
        <button class="prompt-chip" data-prompt="I've been feeling really anxious lately">Feeling anxious</button>
        <button class="prompt-chip" data-prompt="How can I manage stress at work?">Managing stress</button>
        <button class="prompt-chip" data-prompt="I'm having trouble sleeping because of my worries">Trouble sleeping</button>
        <button class="prompt-chip" data-prompt="What are some coping strategies for depression?">Coping strategies</button>
      </div>
    </div>`;
});

// ── Chat logic ──

function removeWelcome() {
  const welcome = chatArea.querySelector(".welcome");
  if (welcome) welcome.remove();
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;

  const avatarLabel = role === "user" ? "You" : "S";
  div.innerHTML = `
    <div class="avatar">${avatarLabel}</div>
    <div class="bubble">${escapeHtml(text)}</div>`;

  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

let messageId = 0;
const botResourcesMap = new Map();

function addBotMessage(text, userMessage, resources = []) {
  const id = ++messageId;
  if (resources && resources.length > 0) {
    botResourcesMap.set(id, resources);
  }
  const div = document.createElement("div");
  div.className = "message bot";
  div.dataset.id = id;

  let htmlContent = marked.parse(text);
  // Replace citations [1], [2], etc. with custom interactive buttons
  htmlContent = htmlContent.replace(/\[([0-9]+)\]/g, (match, num) => {
    const idx = parseInt(num) - 1;
    return `<button class="citation-btn" data-idx="${idx}">[${num}]</button>`;
  });

  let sourcesHtml = "";
  if (resources && resources.length > 0) {
    sourcesHtml = `
      <div class="sources-list">
        <span class="sources-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Sources:
        </span>
        ${resources.map((res, index) => {
          const scoreVal = typeof res.score === 'number'
            ? (res.score <= 1.0 ? Math.round(res.score * 100) : Math.round(res.score))
            : 100;
          return `<button class="source-chip" data-idx="${index}">[${index + 1}] Match ${scoreVal}%</button>`;
        }).join("")}
      </div>
    `;
  }

  div.innerHTML = `
    <div class="avatar">S</div>
    <div class="bubble-wrap">
      <div class="bubble markdown">${htmlContent}</div>
      ${sourcesHtml}
      <div class="feedback-btns" data-id="${id}">
        <button class="fb-btn" data-vote="up" title="Helpful">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
        </button>
        <button class="fb-btn" data-vote="down" title="Not helpful">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>
        </button>
      </div>
    </div>`;

  div.querySelectorAll(".fb-btn").forEach((btn) => {
    btn.addEventListener("click", () => sendFeedback(btn, userMessage, text));
  });

  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

async function sendFeedback(btn, userMessage, botResponse) {
  const wrap = btn.closest(".feedback-btns");
  if (wrap.classList.contains("voted")) return;

  const vote = btn.dataset.vote;
  wrap.classList.add("voted");
  btn.classList.add("selected");

  try {
    await fetch(proxyUrl(settings.apiUrl + "/feedback"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vote,
        user_message: userMessage,
        bot_response: botResponse,
      }),
    });
  } catch { }
}

function addError(text) {
  const div = document.createElement("div");
  div.className = "message bot";
  div.innerHTML = `
    <div class="avatar">S</div>
    <div class="bubble error-bubble">${escapeHtml(text)}</div>`;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function showTyping() {
  const div = document.createElement("div");
  div.className = "message bot";
  div.id = "typing";
  div.innerHTML = `
    <div class="avatar">S</div>
    <div class="bubble">
      <div class="typing-indicator"><span></span><span></span><span></span></div>
    </div>`;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById("typing");
  if (el) el.remove();
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;

  removeWelcome();
  addMessage("user", text);

  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;
  input.focus();

  showTyping();

  try {
    const url = proxyUrl(settings.apiUrl + settings.endpoint);
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    hideTyping();

    if (!res.ok) {
      if (res.status === 429) {
        throw new Error(
          "You're sending messages too quickly. Please wait a moment and try again."
        );
      }
      const errText = await res.text().catch(() => "");
      throw new Error(
        `Server returned ${res.status}${errText ? ": " + errText : ""}`
      );
    }

    const data = await res.json();

    const reply =
      data.response || data.answer || data.message || data.reply || data.text;
    const resources = data.resources || [];

    addBotMessage(reply || JSON.stringify(data, null, 2), text, resources);
  } catch (err) {
    hideTyping();
    if (err.name === "TypeError" && err.message === "Failed to fetch") {
      addError(
        `Could not connect to ${settings.apiUrl}. Make sure your backend is running and CORS is enabled.`
      );
    } else {
      addError(err.message);
    }
  }
}

// ── Retrieval Document Drawer Logic ──
const docPanel = document.getElementById("doc-panel");
const docPanelOverlay = document.getElementById("doc-panel-overlay");
const docPanelClose = document.getElementById("doc-panel-close");
const docPanelTitle = document.getElementById("doc-panel-title");
const docPanelScoreValue = document.getElementById("doc-panel-score-value");
const docPanelContext = document.getElementById("doc-panel-context");
const docPanelAdvice = document.getElementById("doc-panel-advice");

function showDocumentDetails(num, resource) {
  docPanelTitle.textContent = `Retrieved Source [${num}]`;

  // Format score as percentage if it's less than or equal to 1, or keep as is.
  const scoreVal = typeof resource.score === 'number'
    ? (resource.score <= 1.0 ? Math.round(resource.score * 100) : Math.round(resource.score))
    : 100;
  docPanelScoreValue.textContent = `${scoreVal}% Relevance Match`;

  docPanelContext.textContent = resource.page_content || "No context provided.";
  docPanelAdvice.textContent = resource.response || "No advice provided.";

  docPanel.classList.add("open");
}

function closeDocPanel() {
  docPanel.classList.remove("open");
}

docPanelOverlay.addEventListener("click", closeDocPanel);
docPanelClose.addEventListener("click", closeDocPanel);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeDocPanel();
  }
});

// Event delegation for citation clicks and source menu chip clicks in the chat area
chatArea.addEventListener("click", (e) => {
  const btn = e.target.closest(".citation-btn, .source-chip");
  if (!btn) return;

  const msgDiv = btn.closest(".message.bot");
  if (!msgDiv) return;

  const msgId = parseInt(msgDiv.dataset.id);
  const resources = botResourcesMap.get(msgId);
  if (!resources) return;

  const docIdx = parseInt(btn.dataset.idx);
  const resource = resources[docIdx];
  if (resource) {
    showDocumentDetails(docIdx + 1, resource);
  }
});

