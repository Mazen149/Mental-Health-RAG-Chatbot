from fastapi.testclient import TestClient
from unittest.mock import patch

from src.app import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint (Happy Path)."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_health_check():
    """Test the root endpoint (Happy Path) maps to health check."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("src.app.route_query")
@patch("src.app.rag")
def test_chat_happy_path(mock_rag, mock_route_query):
    """Test the /chat endpoint with a valid query (Happy Path)."""
    # Mock the return value of route_query
    mock_route_query.return_value = {
        "answer": "Hello, I am here to help.",
        "resources": [],
        "language": "English",
        "emotion": ["Joy"],
        "intent": "greeting",
    }

    response = client.post("/chat", json={"query": "hello", "history": []})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Hello, I am here to help."
    assert data["language"] == "English"
    assert "resources" in data


def test_chat_missing_query():
    """Test the /chat endpoint with an empty query (Error Case)."""
    response = client.post("/chat", json={"query": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Query text is required."


def test_chat_history():
    """Test the /chat/history endpoint (Happy Path)."""
    response = client.get("/chat/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_chat_clear():
    """Test the /chat/clear endpoint (Happy Path)."""
    response = client.post("/chat/clear")
    assert response.status_code == 200
    assert response.json()["message"] == "Chat history cleared successfully."


def test_feedback_happy_path():
    """Test the /feedback endpoint with a valid vote (Happy Path)."""
    payload = {
        "vote": "up",
        "user_message": "test message",
        "bot_response": "test response",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_feedback_invalid_vote():
    """Test the /feedback endpoint with an invalid vote (Error Case)."""
    payload = {
        "vote": "maybe",
        "user_message": "test message",
        "bot_response": "test response",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Vote must be 'up' or 'down'."


@patch("src.app.config.GROQ_API_KEY", "")
def test_transcribe_missing_api_key():
    """Test the /transcribe endpoint when API key is missing (Error Case)."""
    # Create a dummy file to upload
    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}
    response = client.post("/transcribe", files=files)
    assert response.status_code == 503
    assert "Groq API key is not configured" in response.json()["detail"]
