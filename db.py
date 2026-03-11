"""SQLite persistence for chats, messages, and session-resume IDs."""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone


DB_PATH = os.path.join(os.getcwd(), "data", "store.db")


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def get_db():
    """Yield a SQLite connection that auto-commits / rolls back and always closes."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist. Call once at startup."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_ids (
                chat_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Chats ─────────────────────────────────────────────────────────────────────

def list_chats() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM chats ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def create_chat(title: str | None = None) -> dict:
    chat = {"id": str(uuid.uuid4()), "title": title or "New Chat", "created_at": _now()}
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
            (chat["id"], chat["title"], chat["created_at"]),
        )
    return chat


def get_chat(chat_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, title, created_at FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_chat(chat_id: str) -> bool:
    """Delete a chat and cascade-delete its messages + session_id. Returns False if not found."""
    with get_db() as conn:
        row = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        return True


# ── Messages ──────────────────────────────────────────────────────────────────

def get_messages(chat_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE chat_id = ? ORDER BY created_at",
            (chat_id,),
        ).fetchall()
        result = []
        for r in rows:
            msg = dict(r)
            msg["content"] = json.loads(msg["content"])
            result.append(msg)
        return result


def add_message(chat_id: str, role: str, content) -> None:
    """Persist a message. `content` is a str (user) or list[dict] (assistant)."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (id, chat_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), chat_id, role, json.dumps(content), _now()),
        )


# ── Session IDs (for Claude SDK resume) ──────────────────────────────────────

def get_session_id(chat_id: str) -> str | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT session_id FROM session_ids WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return row["session_id"] if row else None


def set_session_id(chat_id: str, session_id: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO session_ids (chat_id, session_id) VALUES (?, ?)",
            (chat_id, session_id),
        )
