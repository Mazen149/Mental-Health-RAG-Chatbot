# Sanad AI — Empathetic Mental Health Support Chatbot Frontend

A premium, highly interactive, and responsive web chat interface built for **Sanad AI** — a multi-layered mental health support agent backed by hybrid retrieval (BM25 + Qdrant), speech transcription, multilingual classification models, and empathetic LLM grounding.

---

## ✨ Features

### 1. 🔑 Security Portal & Authentication
- Centered **Login & Registration** cards styled in deep dark layouts matching provided guidelines.
- Standard username and password inputs, plus confirm password checks for new account creations.
- Integrates with FastAPI cookie sessions (`SessionMiddleware`) using secure credential management (`credentials: "include"`).
- Automatically tracks and greets active users in the header panel (e.g., `WELCOME BACK, D`).

### 2. 💬 Real-Time SSE Chat Streaming
- Connects to the `/chat/stream` endpoint to stream empathetic responses word-by-word.
- Utilizes browser `ReadableStream` reader pipelines for optimal performance.
- Automatically handles markdown rendering and citation replacements.
- Retains backward-compatible normal POST mode for regular endpoints like `/chat`.

### 3. 🎙️ Integrated Voice Recording & Transcription
- Quick recording icon (`btn-mic`) inside the capsule-shaped messaging input.
- Relies on the standard HTML5 `MediaRecorder` API to record voice messages (WebM, OGG, or MP4 based on browser support).
- Visual recording feedback: pulsating recording state and active timer directly inside the capsule.
- Posts recorded audio to the `/transcribe` endpoint (powered by Whisper Large v3) and populates the query input box instantly.

### 4. 👍 Like & Dislike Feedback
- Floating thumbs up / thumbs down button icons displayed neatly below bot responses.
- Allows users to submit helpfulness ratings to the `/feedback` endpoint.

### 5. 📂 Retrieval Source Details Drawer
- Clickable citation markers `[1]` and Match score chips directly inside bot replies.
- Slides open a details drawer panel showing:
  - **Counseling Case Context**: The original clinical case study that was matched.
  - **Clinical Advice / Response**: The verified clinician response mapped to that case.
- *Note: Emotion, intent, and language tags are processed on the backend but intentionally hidden from the frontend layout to keep the user interface minimal.*

### 🧹 Session History Clear
- Keeps chat history loaded across page refreshes by querying `/chat/history`.
- Toggles a DELETE request to `/chat/clear` to scrub interaction records on the server database.

---

## ⚙️ Configuration & Connection Settings

Click the **Gear Icon ⚙️** in the header (or click **API Settings** at the bottom of the login card) to customize backend options:
- **Backend API URL**: Specify the host IP/Domain of the FastAPI backend (e.g., `http://localhost:8000` or production URL).
- **Chat Endpoint**: Target endpoint (defaults to `/chat/stream`).

Settings are persisted in the browser's `localStorage`.

---

## 🚀 Docker Deployment

An optimized production image is built and available on Docker Hub.

### 1. Pull the Image
```bash
docker pull mazen1393/sanad-ai-frontend:latest
```

### 2. Run Container Locally
To run the container mapping host port 80 to container port 80:
```bash
docker run -d -p 80:80 --name sanad-ai-frontend mazen1393/sanad-ai-frontend:latest
```

---

## 🛠️ Project Structure
- `index.html` — Layout and document details drawer.
- `style.css` — Modern design system styling, fonts, inputs, header, logout layout, and capsule input shapes.
- `app.js` — Core JavaScript logic (SSE streams, audio recording, login/registration actions, settings, and drawer management).
- `nginx.conf` — Web server config for production hosting.
