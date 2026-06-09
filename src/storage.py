from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "serene_ai.sqlite3"
SESSION_TTL_DAYS = 7
PBKDF2_ITERATIONS = 200_000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    return _utc_now().isoformat()


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                language TEXT,
                emotion_json TEXT,
                intent TEXT,
                history_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """
        )


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt_bytes = salt or secrets.token_bytes(16)
    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        PBKDF2_ITERATIONS,
    )
    return (
        base64.b64encode(derived_key).decode("ascii"),
        base64.b64encode(salt_bytes).decode("ascii"),
    )


def _verify_password(password: str, password_hash: str, password_salt: str) -> bool:
    recalculated_hash, _ = _hash_password(password, base64.b64decode(password_salt))
    return hmac.compare_digest(recalculated_hash, password_hash)


def create_user(username: str, password: str) -> dict[str, Any]:
    clean_username = username.strip()
    if not clean_username or not password:
        raise ValueError("Username and password are required.")
    if len(clean_username) < 3:
        raise ValueError("Username must be at least 3 characters long.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    password_hash, password_salt = _hash_password(password)
    created_at = _utc_iso()

    try:
        with _connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, password_hash, password_salt, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean_username, password_hash, password_salt, created_at),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise ValueError("That username is already registered.") from exc

    return {
        "id": user_id,
        "username": clean_username,
        "created_at": created_at,
    }


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    clean_username = username.strip()
    if not clean_username or not password:
        return None

    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, username, password_hash, password_salt, created_at
            FROM users
            WHERE username = ?
            """,
            (clean_username,),
        ).fetchone()

    if row is None:
        return None
    if not _verify_password(password, row["password_hash"], row["password_salt"]):
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
    }


def create_session(user_id: int) -> dict[str, str]:
    token = secrets.token_urlsafe(32)
    created_at = _utc_iso()
    expires_at = (_utc_now() + timedelta(days=SESSION_TTL_DAYS)).isoformat()

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO sessions (token, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (token, user_id, created_at, expires_at),
        )

    return {"token": token, "expires_at": expires_at}


def get_user_by_session(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None

    now = _utc_iso()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT u.id, u.username, u.created_at, s.expires_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ? AND s.expires_at > ?
            """,
            (token, now),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
        "expires_at": row["expires_at"],
    }


def revoke_session(token: str | None) -> None:
    if not token:
        return

    with _connect() as connection:
        connection.execute("DELETE FROM sessions WHERE token = ?", (token,))


def record_interaction(
    user_id: int,
    *,
    query: str,
    answer: str,
    language: str | None = None,
    emotion: list[str] | None = None,
    intent: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> None:
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO interactions (
                user_id, query, answer, language, emotion_json, intent, history_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                query,
                answer,
                language,
                json.dumps(emotion, ensure_ascii=False) if emotion is not None else None,
                intent,
                json.dumps(history, ensure_ascii=False) if history is not None else None,
                _utc_iso(),
            ),
        )
