import sys
import sqlite3
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class TestFeedback(unittest.TestCase):
    def setUp(self):
        # Create temp dir for sqlite database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_chat.sqlite3"

        # Patch the database path in src.app
        self.db_patcher = patch("src.app.DB_PATH", self.db_path)
        self.db_patcher.start()

        # Patch config to disable Turso connection during tests
        self.config_patcher = patch("src.app.config.TURSO_DATABASE_URL", None)
        self.config_patcher.start()

        # Initialize the test DB schema
        from src.app import _init_chat_db

        _init_chat_db()

        # Import app
        from src.app import app

        self.client = TestClient(app)

        # Bypass auth by injecting user_id and returning True
        def mock_auth(req):
            req.session["authenticated"] = True
            req.session["user_id"] = 1
            return True

        # Robust authentication bypass: override _is_authenticated in loaded app modules specifically
        self.original_funcs = {}
        for name, module in list(sys.modules.items()):
            if (
                module is not None
                and (name == "src.app" or name == "app")
                and hasattr(module, "_is_authenticated")
            ):
                self.original_funcs[name] = module._is_authenticated
                module._is_authenticated = mock_auth

    def tearDown(self):
        # Restore original functions
        for name, original_func in self.original_funcs.items():
            if name in sys.modules and sys.modules[name] is not None:
                sys.modules[name]._is_authenticated = original_func

        self.db_patcher.stop()
        self.config_patcher.stop()

        # Clean up temp database directory
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass

    @patch("src.app.rag")
    @patch("src.app.route_query")
    def test_chat_with_query_payload(self, mock_route, mock_rag):
        """Test POST /chat with { 'query': '...' }."""
        mock_route.return_value = {
            "answer": "This is a mock answer.",
            "resources": [],
            "language": "English",
            "emotion": ["Joy"],
            "intent": "greeting",
        }

        response = self.client.post("/chat", json={"query": "hello there"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "This is a mock answer.")
        mock_route.assert_called_once_with("hello there", mock_rag, history=None)

    @patch("src.app.rag")
    @patch("src.app.route_query")
    def test_chat_with_message_payload(self, mock_route, mock_rag):
        """Test POST /chat with { 'message': '...' } as fallback."""
        mock_route.return_value = {
            "answer": "This is another mock answer.",
            "resources": [],
            "language": "English",
            "emotion": ["Joy"],
            "intent": "greeting",
        }

        response = self.client.post("/chat", json={"message": "hello message field"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "This is another mock answer.")
        mock_route.assert_called_once_with(
            "hello message field", mock_rag, history=None
        )

    def test_feedback_submission_success(self):
        """Test successful feedback submission with up/down votes."""
        # Thumbs up
        response_up = self.client.post(
            "/feedback",
            json={
                "vote": "up",
                "user_message": "User query",
                "bot_response": "Bot reply",
            },
        )
        self.assertEqual(response_up.status_code, 200)
        self.assertEqual(
            response_up.json(),
            {"status": "ok", "message": "Feedback saved successfully."},
        )

        # Thumbs down
        response_down = self.client.post(
            "/feedback",
            json={
                "vote": "down",
                "user_message": "User query",
                "bot_response": "Bot reply",
            },
        )
        self.assertEqual(response_down.status_code, 200)
        self.assertEqual(
            response_down.json(),
            {"status": "ok", "message": "Feedback saved successfully."},
        )

        # Query the database directly to verify persistence
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT vote, user_message, bot_response FROM feedback ORDER BY id ASC"
            )
            rows = cursor.fetchall()

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0], ("up", "User query", "Bot reply"))
            self.assertEqual(rows[1], ("down", "User query", "Bot reply"))

    def test_feedback_invalid_vote(self):
        """Test that invalid votes return a 400 bad request error."""
        response = self.client.post(
            "/feedback",
            json={"vote": "maybe", "user_message": "hello", "bot_response": "world"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Vote must be 'up' or 'down'", response.json()["detail"])

    def test_feedback_unauthenticated_uses_guest_user(self):
        """Test that unauthenticated requests to /feedback fall back to the guest user."""
        # Restore original auth functions to simulate unauthenticated request
        for name, original_func in self.original_funcs.items():
            if name in sys.modules and sys.modules[name] is not None:
                sys.modules[name]._is_authenticated = original_func

        # Use a client without mock authentication session variables
        from src.app import app

        unauth_client = TestClient(app)

        response = unauth_client.post(
            "/feedback",
            json={
                "vote": "up",
                "user_message": "guest query",
                "bot_response": "guest response",
            },
        )
        self.assertEqual(response.status_code, 200)

        # Verify that guest user was created and feedback was saved under guest user id
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username FROM users WHERE username = ?", ("guest",)
            )
            guest_user = cursor.fetchone()
            self.assertIsNotNone(guest_user)
            guest_id = guest_user[0]

            cursor.execute(
                "SELECT user_id, vote, user_message FROM feedback WHERE user_message = ?",
                ("guest query",),
            )
            feedback_row = cursor.fetchone()
            self.assertIsNotNone(feedback_row)
            self.assertEqual(feedback_row[0], guest_id)
            self.assertEqual(feedback_row[1], "up")
