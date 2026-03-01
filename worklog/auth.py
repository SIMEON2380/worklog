import sqlite3
from datetime import datetime

import bcrypt
import streamlit as st

from .config import Config

USERS_TABLE = "users"


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _db_path(cfg: Config) -> str:
    # Use the same SQLite file as the rest of the app
    return cfg.DB_PATH


def _connect(cfg: Config) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(cfg), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_default_user(cfg: Config) -> None:
    """
    Creates users table and ensures an admin user exists.
    Seed password priority:
      1) st.secrets["ADMIN_PASSWORD"] if set
      2) "admin123"
    """
    with _connect(cfg) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {USERS_TABLE} (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        row = conn.execute(f"SELECT COUNT(*) AS c FROM {USERS_TABLE}").fetchone()
        if int(row["c"] or 0) > 0:
            return

        seed_pw = None
        try:
            seed_pw = st.secrets.get("ADMIN_PASSWORD")
        except Exception:
            seed_pw = None

        if not seed_pw:
            seed_pw = "admin123"

        pw_hash = bcrypt.hashpw(seed_pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        now = _now()

        conn.execute(
            f"""
            INSERT INTO {USERS_TABLE} (username, password_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            ("admin", pw_hash, now, now),
        )


def verify_login(cfg: Config, username: str, password: str) -> bool:
    username = (username or "").strip()
    if not username:
        return False

    with _connect(cfg) as conn:
        row = conn.execute(
            f"SELECT password_hash FROM {USERS_TABLE} WHERE username = ?",
            (username,),
        ).fetchone()

    if not row:
        return False

    stored = row["password_hash"].encode("utf-8")
    return bool(bcrypt.checkpw(password.encode("utf-8"), stored))


def change_password(cfg: Config, username: str, old_password: str, new_password: str) -> str:
    username = (username or "").strip()
    if not username:
        return "No user."

    if not new_password or len(new_password) < 6:
        return "New password must be at least 6 characters."

    if not verify_login(cfg, username, old_password):
        return "Old password is wrong."

    new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    with _connect(cfg) as conn:
        conn.execute(
            f"UPDATE {USERS_TABLE} SET password_hash = ?, updated_at = ? WHERE username = ?",
            (new_hash, _now(), username),
        )

    return "Password updated."