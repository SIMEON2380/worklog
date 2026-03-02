import sqlite3
import hashlib
import hmac
from typing import Optional, Tuple, Any


def _get_cfg(cfg_or_db: Any):
    # If they passed the DB dict from make_db(cfg), try to recover cfg
    if isinstance(cfg_or_db, dict):
        # common patterns; keep it flexible
        if "cfg" in cfg_or_db:
            return cfg_or_db["cfg"]
        if "__cfg__" in cfg_or_db:
            return cfg_or_db["__cfg__"]
    return cfg_or_db


def _db_path(cfg_or_db: Any) -> str:
    cfg = _get_cfg(cfg_or_db)
    if not hasattr(cfg, "DB_PATH"):
        raise AttributeError(
            "Config must have DB_PATH. Fix worklog/config.py or pass cfg into auth funcs."
        )
    return cfg.DB_PATH


def _connect(cfg_or_db: Any):
    conn = sqlite3.connect(_db_path(cfg_or_db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password: str, salt: str) -> str:
    # simple stable hash (upgrade later if you want)
    data = (salt + password).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _verify_password(password: str, salt: str, stored_hash: str) -> bool:
    cand = _hash_password(password, salt)
    return hmac.compare_digest(cand, stored_hash)


def ensure_default_user(cfg_or_db: Any):
    """
    Ensures users table exists and creates default user if config provides it.
    Expects cfg.DEFAULT_USERNAME and cfg.DEFAULT_PASSWORD if you use defaults.
    If your config uses different names, edit below.
    """
    cfg = _get_cfg(cfg_or_db)

    with _connect(cfg) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()

        default_username = getattr(cfg, "DEFAULT_USERNAME", None)
        default_password = getattr(cfg, "DEFAULT_PASSWORD", None)

        # If you don't want a default user, just skip
        if not default_username or not default_password:
            return

        cur = conn.execute("SELECT 1 FROM users WHERE username = ?", (default_username,))
        exists = cur.fetchone() is not None
        if exists:
            return

        salt = hashlib.sha256(default_username.encode("utf-8")).hexdigest()[:16]
        pw_hash = _hash_password(default_password, salt)
        conn.execute(
            "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
            (default_username, salt, pw_hash),
        )
        conn.commit()


def verify_login(cfg_or_db: Any, username: str, password: str) -> Optional[str]:
    if not username or not password:
        return None

    with _connect(cfg_or_db) as conn:
        row = conn.execute(
            "SELECT username, salt, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if not row:
        return None

    if _verify_password(password, row["salt"], row["password_hash"]):
        return row["username"]

    return None


def change_password(cfg_or_db: Any, username: str, current_password: str, new_password: str) -> Tuple[bool, str]:
    if not username:
        return False, "Not logged in."
    if not current_password or not new_password:
        return False, "Missing password."
    if len(new_password) < 6:
        return False, "New password too short (min 6 chars)."

    with _connect(cfg_or_db) as conn:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if not row:
            return False, "User not found."

        if not _verify_password(current_password, row["salt"], row["password_hash"]):
            return False, "Current password is wrong."

        salt = row["salt"]  # keep existing salt
        pw_hash = _hash_password(new_password, salt)
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (pw_hash, username),
        )
        conn.commit()

    return True, "Password updated."