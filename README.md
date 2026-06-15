# Sanad AI - Mental Health Support Chatbot

A simple chat interface for the Sanad AI mental health support chatbot.

**Live:** https://ishraq-hassan.github.io/chatbot-frontend/

## How it works

The frontend sends user messages to a backend API and displays the responses. It also lets users give thumbs up/down feedback on bot replies.

### Endpoints used

- `POST /chat` with `{ "message": "..." }` - sends a message, gets a response
- `POST /feedback` with `{ "vote": "up|down", "user_message": "...", "bot_response": "..." }` - sends feedback

### Files

- `index.html` - page structure
- `style.css` - styling
- `app.js` - chat logic, settings, API calls

## For students (Final Project)

1. Fork this repo.
2. In `app.js`, update the default `apiUrl` to point to your deployed backend.
3. Push to `main`. GitHub Actions will deploy it to GitHub Pages automatically.
4. If your repo is private, make it public first (GitHub Pages requires this on the free plan).

## Settings

Click the gear icon in the header to change the backend URL and chat endpoint at any time. Settings are saved in your browser's localStorage.
