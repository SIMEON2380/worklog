import os
import sqlite3
from datetime import datetime
from typing import Optional

import bcrypt
import streamlit as st

from .config import Config


USERS_TABLE = "users"


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _users_db_path(cfg: Config) -> str:
    # same DB file, same location
    return cfg.DB_PATH


def ensure_users_table(cfg: Config) -> None:
    conn = sqlite3.connect(_users_db_path(cfg), check_same_thread=False)
    try:
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
        conn.commit()

        # If no users exist, seed admin from secrets if present, else admin123
        row = conn.execute(f"SELECT COUNT(*) FROM {USERS_TABLE}").fetchone()
        if int(row[0] or 0) == 0:
            seed_pw = None
            try:
                seed_pw = st.secrets.get("ADMIN_PASSWORD")
            except Exception:
                seed_pw = None
            if not seed_pw:
                seed_pw = "admin123"

            pw_hash = bcrypt.hashpw(seed_pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            conn.execute(
                f"INSERT INTO {USERS_TABLE} (username, password_hash, created_at, updated_at) VALUES (?, ?, ?, ?)",
                ("admin", pw_hash, _now(), _now()),
            )
            conn.commit()
    finally:
        conn.close()


def verify_admin_password(cfg: Config, password: str) -> bool:
    """
    Checks SQLite user password first.
    If anything fails, fallback to st.secrets["ADMIN_PASSWORD"].
    """
    ensure_users_table(cfg)

    conn = sqlite3.connect(_users_db_path(cfg), check_same_thread=False)
    try:
        row = conn.execute(f"SELECT password_hash FROM {USERS_TABLE} WHERE username = ?", ("admin",)).fetchone()
        if row and row[0]:
            stored = row[0].encode("utf-8")
            return bool(bcrypt.checkpw(password.encode("utf-8"), stored))
    finally:
        conn.close()

    # fallback (old behaviour)
    try:
        return password == st.secrets["ADMIN_PASSWORD"]
    except Exception:
        return False


def change_admin_password(cfg: Config, old_password: str, new_password: str) -> str:
    if not new_password or len(new_password) < 6:
        return "New password must be at least 6 characters."

    if not verify_admin_password(cfg, old_password):
        return "Old password is wrong."

    ensure_users_table(cfg)

    new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    conn = sqlite3.connect(_users_db_path(cfg), check_same_thread=False)
    try:
        conn.execute(
            f"UPDATE {USERS_TABLE} SET password_hash = ?, updated_at = ? WHERE username = ?",
            (new_hash, _now(), "admin"),
        )
        conn.commit()
    finally:
        conn.close()

    return "Password updated."