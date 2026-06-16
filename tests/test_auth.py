import sys
import sqlite3
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class TestAuth(unittest.TestCase):
    def setUp(self):
        # Create temp dir for sqlite database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_auth.sqlite3"

        # Patch the database path in src.app
        self.db_patcher = patch("src.app.DB_PATH", self.db_path)
        self.db_patcher.start()

        # Patch config to disable Turso connection during tests
        self.config_patcher = patch("src.app.config.TURSO_DATABASE_URL", None)
        self.config_patcher.start()

        self.cookie_secure_patcher = patch("src.app.config.SESSION_COOKIE_SECURE", False)
        self.cookie_secure_patcher.start()

        self.cookie_samesite_patcher = patch("src.app.config.SESSION_COOKIE_SAMESITE", "lax")
        self.cookie_samesite_patcher.start()

        # Initialize the test DB schema
        from src.app import _init_chat_db
        _init_chat_db()

        # Import app
        from src.app import app
        self.client = TestClient(app)

    def tearDown(self):
        self.db_patcher.stop()
        self.config_patcher.stop()
        self.cookie_secure_patcher.stop()
        self.cookie_samesite_patcher.stop()

        # Clean up temp database directory
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass

    def test_registration_and_login_flow(self):
        """Test user registration, login, session retention, and logout."""
        # 1. Register a new user
        reg_payload = {"username": "testuser", "password": "securepassword"}
        response = self.client.post("/register", json=reg_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["username"], "testuser")

        # 2. Re-registering the same username should fail
        response_dup = self.client.post("/register", json=reg_payload)
        self.assertEqual(response_dup.status_code, 400)
        self.assertIn("Username already registered", response_dup.json()["detail"])

        # 3. Logout the active session
        response_logout = self.client.post("/logout")
        self.assertEqual(response_logout.status_code, 200)
        self.assertEqual(response_logout.json()["status"], "ok")

        # 4. Login with correct credentials
        login_payload = {"username": "testuser", "password": "securepassword"}
        response_login = self.client.post("/login", json=login_payload)
        self.assertEqual(response_login.status_code, 200)
        data_login = response_login.json()
        self.assertEqual(data_login["status"], "ok")
        self.assertEqual(data_login["username"], "testuser")

        # 5. Login with incorrect password should fail
        bad_login_payload = {"username": "testuser", "password": "wrongpassword"}
        response_bad = self.client.post("/login", json=bad_login_payload)
        self.assertEqual(response_bad.status_code, 401)
        self.assertIn("Incorrect username or password", response_bad.json()["detail"])

    def test_unauthenticated_requests_blocked(self):
        """Test that unauthenticated requests to chat and history endpoints are blocked with 401."""
        # Chat history
        res_history = self.client.get("/chat/history")
        self.assertEqual(res_history.status_code, 401)
        self.assertIn("Authentication required", res_history.json()["detail"])

        # Clear history
        res_clear = self.client.post("/chat/clear")
        self.assertEqual(res_clear.status_code, 401)
        self.assertIn("Authentication required", res_clear.json()["detail"])

        # Send chat
        res_chat = self.client.post("/chat", json={"query": "hello"})
        self.assertEqual(res_chat.status_code, 401)
        self.assertIn("Authentication required", res_chat.json()["detail"])

        # Stream chat
        res_stream = self.client.post("/chat/stream", json={"query": "hello"})
        self.assertEqual(res_stream.status_code, 401)
        self.assertIn("Authentication required", res_stream.json()["detail"])
