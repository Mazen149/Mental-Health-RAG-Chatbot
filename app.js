const chatArea = document.getElementById("chat-area");
const input = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");

// Auth DOM references (New layout matching Image 1 & 2)
const authOverlay = document.getElementById("auth-overlay");
const loginCard = document.getElementById("login-card");
const registerCard = document.getElementById("register-card");
const loginForm = document.getElementById("login-form");
const registerForm = document.getElementById("register-form");

const loginUsernameInput = document.getElementById("login-username");
const loginPasswordInput = document.getElementById("login-password");
const registerUsernameInput = document.getElementById("register-username");
const registerPasswordInput = document.getElementById("register-password");
const registerConfirmPasswordInput = document.getElementById("register-confirm-password");

const loginError = document.querySelector(".login-error");
const registerError = document.querySelector(".register-error");

const goToRegisterBtn = document.getElementById("go-to-register");
const goToLoginBtn = document.getElementById("go-to-login");

const welcomeUser = document.getElementById("welcome-user");
const logoutBtn = document.getElementById("logout-btn");
const micBtn = document.getElementById("mic-btn");
const recordingStatus = document.getElementById("recording-status");
const recordingTimer = document.getElementById("recording-timer");

const STORAGE_KEY = "sanad_ai_settings";
const defaults = {
  apiUrl: window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000" // Local development backend address
    : "https://sanad-ai.myvnc.com",
  endpoint: "/chat/stream" // Defaulting to SSE streaming
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
let currentUser = null;
let chatHistory = [];
const botResourcesMap = new Map();
let messageId = 0;

// Recording state
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let recordingInterval = null;
let recordingSeconds = 0;
let micStream = null;

// ── Session & Auth management ──

async function checkAuth() {
  try {
    const url = proxyUrl(settings.apiUrl + "/chat/history");
    const res = await fetch(url, { credentials: "include" });
    if (res.ok) {
      currentUser = localStorage.getItem("sanad_username") || "User";
      authOverlay.classList.add("hidden");
      welcomeUser.textContent = currentUser.toUpperCase();
      await loadHistory();
    } else {
      authOverlay.classList.remove("hidden");
    }
  } catch (err) {
    console.error("Auth check failed:", err);
    authOverlay.classList.remove("hidden");
    loginError.textContent = `Could not connect to backend at ${settings.apiUrl || "local base"}. Please ensure your backend server is running and check settings.`;
    loginError.style.display = "block";
  }
}

async function loadHistory() {
  try {
    const url = proxyUrl(settings.apiUrl + "/chat/history");
    const res = await fetch(url, { credentials: "include" });
    if (!res.ok) {
      if (res.status === 401) {
        authOverlay.classList.remove("hidden");
      }
      return;
    }
    
    const data = await res.json();
    chatArea.innerHTML = "";
    chatHistory = [];
    botResourcesMap.clear();
    
    if (data && data.length > 0) {
      removeWelcome();
      data.forEach(item => {
        chatHistory.push({ role: item.role, content: item.content });
        if (item.role === "user") {
          addMessage("user", item.content);
        } else {
          addBotMessage(
            item.content,
            chatHistory.length >= 2 ? chatHistory[chatHistory.length - 2].content : "",
            item.resources || []
          );
        }
      });
    } else {
      resetWelcome();
    }
  } catch (err) {
    console.error("Error loading chat history:", err);
  }
}

// ── Card Toggling ──

goToRegisterBtn.addEventListener("click", () => {
  loginCard.classList.add("hidden");
  registerCard.classList.remove("hidden");
  registerError.style.display = "none";
});

goToLoginBtn.addEventListener("click", () => {
  registerCard.classList.add("hidden");
  loginCard.classList.remove("hidden");
  loginError.style.display = "none";
});


// ── Login / Register Form Submissions ──

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = loginUsernameInput.value.trim();
  const password = loginPasswordInput.value.trim();
  
  if (!username || !password) return;
  
  const submitBtn = loginForm.querySelector(".btn-auth-submit");
  submitBtn.disabled = true;
  loginError.style.display = "none";
  
  const url = proxyUrl(settings.apiUrl + "/login");
  
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
      credentials: "include"
    });
    
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Server returned ${res.status}`);
    }
    
    currentUser = data.username || username;
    localStorage.setItem("sanad_username", currentUser);
    localStorage.setItem("sanad_logged_in", "true");
    
    authOverlay.classList.add("hidden");
    welcomeUser.textContent = currentUser.toUpperCase();
    
    await loadHistory();
    
    loginUsernameInput.value = "";
    loginPasswordInput.value = "";
  } catch (err) {
    loginError.textContent = err.message;
    loginError.style.display = "block";
  } finally {
    submitBtn.disabled = false;
  }
});

registerForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = registerUsernameInput.value.trim();
  const password = registerPasswordInput.value.trim();
  const confirmPassword = registerConfirmPasswordInput.value.trim();
  
  if (!username || !password || !confirmPassword) return;
  
  if (password !== confirmPassword) {
    registerError.textContent = "Passwords do not match.";
    registerError.style.display = "block";
    return;
  }
  
  const submitBtn = registerForm.querySelector(".btn-auth-submit");
  submitBtn.disabled = true;
  registerError.style.display = "none";
  
  const url = proxyUrl(settings.apiUrl + "/register");
  
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
      credentials: "include"
    });
    
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Server returned ${res.status}`);
    }
    
    currentUser = data.username || username;
    localStorage.setItem("sanad_username", currentUser);
    localStorage.setItem("sanad_logged_in", "true");
    
    authOverlay.classList.add("hidden");
    welcomeUser.textContent = currentUser.toUpperCase();
    
    await loadHistory();
    
    registerUsernameInput.value = "";
    registerPasswordInput.value = "";
    registerConfirmPasswordInput.value = "";
    
    // Auto switch card state back to login for future sessions
    registerCard.classList.add("hidden");
    loginCard.classList.remove("hidden");
  } catch (err) {
    registerError.textContent = err.message;
    registerError.style.display = "block";
  } finally {
    submitBtn.disabled = false;
  }
});

logoutBtn.addEventListener("click", async () => {
  try {
    await fetch(proxyUrl(settings.apiUrl + "/logout"), {
      method: "POST",
      credentials: "include"
    });
  } catch (err) {
    console.error("Logout request failed:", err);
  }
  
  currentUser = null;
  localStorage.removeItem("sanad_username");
  localStorage.removeItem("sanad_logged_in");
  authOverlay.classList.remove("hidden");
  chatHistory = [];
  botResourcesMap.clear();
  resetWelcome();
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

clearBtn.addEventListener("click", async () => {
  if (confirm("Are you sure you want to clear your chat history?")) {
    try {
      await fetch(proxyUrl(settings.apiUrl + "/chat/clear"), {
        method: "POST",
        credentials: "include"
      });
    } catch (err) {
      console.error("Failed to clear chat on server:", err);
    }
    chatHistory = [];
    botResourcesMap.clear();
    resetWelcome();
  }
});

function resetWelcome() {
  chatArea.innerHTML = `
    <div class="welcome" id="welcome-block">
      <h2>Welcome to Sanad AI</h2>
      <p>I am here to listen and support you. You can ask me questions about anxiety, depression, stress, relationships, or anything else on your mind. All my answers are grounded in trusted counseling guidelines.</p>
      
      <div class="quick-prompts">
        <button class="prompt-chip" data-prompt="I've been feeling really anxious lately">Feeling anxious</button>
        <button class="prompt-chip" data-prompt="How can I manage stress at work?">Managing stress</button>
        <button class="prompt-chip" data-prompt="I'm having trouble sleeping because of my worries">Trouble sleeping</button>
        <button class="prompt-chip" data-prompt="What are some coping strategies for depression?">Coping strategies</button>
      </div>
    </div>`;
}

// ── Chat logic ──

function removeWelcome() {
  const welcome = chatArea.querySelector(".welcome");
  if (welcome) welcome.remove();
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;

  const username = localStorage.getItem("sanad_username") || "User";
  const initial = username.charAt(0).toUpperCase();

  div.innerHTML = `
    <div class="avatar user-avatar">${initial}</div>
    <div class="bubble">${escapeHtml(text)}</div>`;

  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

function addBotMessage(text, userMessage, resources = []) {
  const id = ++messageId;
  if (resources && resources.length > 0) {
    botResourcesMap.set(id, resources);
  }
  const div = document.createElement("div");
  div.className = "message bot";
  div.dataset.id = id;

  let htmlContent = marked.parse(text);
  // Replace citations [1], [2], etc.
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
          return `<button class="source-chip" data-idx="${index}"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="source-chip-icon"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><span class="source-idx">Source ${index + 1}</span><span class="source-match">${scoreVal}% Match</span></button>`;
        }).join("")}
      </div>
    `;
  }

  div.innerHTML = `
    <div class="avatar bot-avatar">🤖</div>
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
  const vote = btn.dataset.vote;

  const isAlreadySelected = btn.classList.contains("selected");

  if (isAlreadySelected) {
    // Undo selection on UI
    btn.classList.remove("selected");
    wrap.classList.remove("voted");
    return;
  }

  // Clear existing selections and toggle new vote
  wrap.querySelectorAll(".fb-btn").forEach(b => b.classList.remove("selected"));
  btn.classList.add("selected");
  wrap.classList.add("voted");

  try {
    await fetch(proxyUrl(settings.apiUrl + "/feedback"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vote,
        user_message: userMessage,
        bot_response: botResponse,
      }),
      credentials: "include"
    });
  } catch (err) {
    console.error("Failed to send feedback:", err);
  }
}

function addError(text) {
  const div = document.createElement("div");
  div.className = "message bot";
  div.innerHTML = `
    <div class="bubble error-bubble">${escapeHtml(text)}</div>`;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function showTyping(message = "") {
  const div = document.createElement("div");
  div.className = "message bot";
  div.id = "typing";
  div.innerHTML = `
    <div class="avatar bot-avatar">🤖</div>
    <div class="bubble-wrap">
      <div class="bubble">
        ${message ? `<div style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:6px;font-weight:500;">${escapeHtml(message)}</div>` : ""}
        <div class="typing-indicator"><span></span><span></span><span></span></div>
      </div>
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

function renderSourcesList(container, resources) {
  if (!resources || resources.length === 0) return;
  container.innerHTML = `
    <span class="sources-title">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
      Sources:
    </span>
    ${resources.map((res, index) => {
      const scoreVal = typeof res.score === 'number'
        ? (res.score <= 1.0 ? Math.round(res.score * 100) : Math.round(res.score))
        : 100;
      return `<button class="source-chip" data-idx="${index}"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="source-chip-icon"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><span class="source-idx">Source ${index + 1}</span><span class="source-match">${scoreVal}% Match</span></button>`;
    }).join("")}
  `;
  container.style.display = "flex";
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

  const isStreaming = settings.endpoint.includes("/stream");

  try {
    const url = proxyUrl(settings.apiUrl + settings.endpoint);

    if (isStreaming) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: chatHistory.map(h => ({ role: h.role, content: h.content }))
        }),
        credentials: "include"
      });

      hideTyping();

      if (!response.ok) {
        const errText = await response.text().catch(() => "");
        throw new Error(`Server returned ${response.status}${errText ? ": " + errText : ""}`);
      }

      // Prepare bot message bubble
      const botMsgId = ++messageId;
      const botDiv = document.createElement("div");
      botDiv.className = "message bot";
      botDiv.dataset.id = botMsgId;

      botDiv.innerHTML = `
        <div class="avatar bot-avatar">🤖</div>
        <div class="bubble-wrap">
          <div class="bubble markdown"></div>
          <div class="sources-list" style="display: none;"></div>
          <div class="feedback-btns" data-id="${botMsgId}" style="display: none;">
            <button class="fb-btn" data-vote="up" title="Helpful">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
            </button>
            <button class="fb-btn" data-vote="down" title="Not helpful">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>
            </button>
          </div>
        </div>`;

      chatArea.appendChild(botDiv);
      chatArea.scrollTop = chatArea.scrollHeight;

      const bubbleEl = botDiv.querySelector(".bubble");
      const sourcesEl = botDiv.querySelector(".sources-list");
      const feedbackEl = botDiv.querySelector(".feedback-btns");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let partialData = "";
      let accumulatedAnswer = "";
      let streamResources = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        partialData += decoder.decode(value, { stream: true });

        const lines = partialData.split("\n");
        partialData = lines.pop(); // Save incomplete lines

        let currentEvent = "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith("event:")) {
            currentEvent = trimmed.replace("event:", "").trim();
          } else if (trimmed.startsWith("data:")) {
            const dataStr = trimmed.replace("data:", "").trim();
            try {
              const parsed = JSON.parse(dataStr);
              if (currentEvent === "chunk") {
                accumulatedAnswer += parsed.text + " ";
                bubbleEl.innerHTML = marked.parse(accumulatedAnswer.trim());
                bubbleEl.innerHTML = bubbleEl.innerHTML.replace(/\[([0-9]+)\]/g, (match, num) => {
                  const idx = parseInt(num) - 1;
                  return `<button class="citation-btn" data-idx="${idx}">[${num}]</button>`;
                });
                chatArea.scrollTop = chatArea.scrollHeight;
              } else if (currentEvent === "citations") {
                streamResources = parsed.resources || [];
                if (streamResources.length > 0) {
                  botResourcesMap.set(botMsgId, streamResources);
                  renderSourcesList(sourcesEl, streamResources);
                }
              }
            } catch (e) {
              if (currentEvent === "chunk") {
                accumulatedAnswer += dataStr + " ";
                bubbleEl.innerHTML = marked.parse(accumulatedAnswer.trim());
                chatArea.scrollTop = chatArea.scrollHeight;
              }
            }
          }
        }
      }

      feedbackEl.style.display = "flex";
      feedbackEl.querySelectorAll(".fb-btn").forEach((btn) => {
        btn.addEventListener("click", () => sendFeedback(btn, text, accumulatedAnswer.trim()));
      });

      chatHistory.push({ role: "user", content: text });
      chatHistory.push({ role: "assistant", content: accumulatedAnswer.trim() });

    } else {
      // Non-streaming fallback
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: chatHistory.map(h => ({ role: h.role, content: h.content }))
        }),
        credentials: "include"
      });

      hideTyping();

      if (!res.ok) {
        if (res.status === 429) {
          throw new Error("You're sending messages too quickly. Please wait a moment and try again.");
        }
        const errText = await res.text().catch(() => "");
        throw new Error(`Server returned ${res.status}${errText ? ": " + errText : ""}`);
      }

      const data = await res.json();
      const reply = data.answer || data.response || data.message || data.reply || data.text || "";
      const resources = data.resources || [];

      addBotMessage(reply || JSON.stringify(data, null, 2), text, resources);

      chatHistory.push({ role: "user", content: text });
      chatHistory.push({ role: "assistant", content: reply });
    }

  } catch (err) {
    hideTyping();
    if (err.name === "TypeError" && err.message === "Failed to fetch") {
      addError(`Could not connect to backend at ${settings.apiUrl}. Please verify the URL is correct and the backend is running.`);
    } else {
      addError(err.message);
    }
  }
}

// ── Voice Transcription / MediaRecorder ──

async function startRecording() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Voice recording is not supported in your browser.");
      return;
    }
    
    audioChunks = [];
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    let options = { mimeType: "audio/webm" };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
      options = { mimeType: "audio/ogg" };
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options = { mimeType: "audio/mp4" };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
          options = { mimeType: "" };
        }
      }
    }
    
    mediaRecorder = new MediaRecorder(micStream, options);
    mediaRecorder.addEventListener("dataavailable", (e) => {
      if (e.data.size > 0) {
        audioChunks.push(e.data);
      }
    });
    
    mediaRecorder.addEventListener("stop", async () => {
      const mimeType = mediaRecorder.mimeType || "audio/wav";
      const audioBlob = new Blob(audioChunks, { type: mimeType });
      
      if (micStream) {
        micStream.getTracks().forEach(track => track.stop());
      }
      
      await transcribeAudio(audioBlob);
    });
    
    mediaRecorder.start();
    isRecording = true;
    micBtn.classList.add("recording");
    
    input.style.display = "none";
    recordingStatus.style.display = "flex";
    
    recordingSeconds = 0;
    recordingTimer.textContent = "0s";
    recordingInterval = setInterval(() => {
      recordingSeconds++;
      recordingTimer.textContent = `${recordingSeconds}s`;
      if (recordingSeconds >= 30) {
        stopRecording();
      }
    }, 1000);
    
  } catch (err) {
    console.error("Microphone recording start failed:", err);
    alert("Could not access microphone: " + err.message);
  }
}

function stopRecording() {
  if (!isRecording) return;
  
  clearInterval(recordingInterval);
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  
  isRecording = false;
  micBtn.classList.remove("recording");
  
  input.style.display = "block";
  recordingStatus.style.display = "none";
}

async function transcribeAudio(audioBlob) {
  showTyping("Transcribing your voice...");
  
  try {
    const formData = new FormData();
    const ext = audioBlob.type.split("/")[1]?.split(";")[0] || "webm";
    formData.append("file", audioBlob, `audio.${ext}`);
    
    const url = proxyUrl(settings.apiUrl + "/transcribe");
    const res = await fetch(url, {
      method: "POST",
      body: formData,
      credentials: "include"
    });
    
    hideTyping();
    
    if (!res.ok) {
      throw new Error(`Failed to transcribe: ${res.statusText}`);
    }
    
    const data = await res.json();
    if (data && data.text) {
      input.value = data.text;
      input.dispatchEvent(new Event("input"));
      input.focus();
    }
  } catch (err) {
    hideTyping();
    addError("Voice Transcription Error: " + err.message);
  }
}

micBtn.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

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

// ── Initialize App ──
window.addEventListener("DOMContentLoaded", () => {
  checkAuth();
});
