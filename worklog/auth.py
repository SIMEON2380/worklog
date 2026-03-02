# worklog/auth.py
import sqlite3
import hashlib
import hmac
from datetime import datetime
from typing import Any, Optional, Tuple

import bcrypt


def _utc_now_str() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _db_path(cfg: Any) -> str:
    # Config has DB_PATH as a @property
    if not hasattr(cfg, "DB_PATH"):
        raise AttributeError(
            "verify_login/change_password must be called with Config (cfg), not the DB dict."
        )
    return cfg.DB_PATH


def _connect(cfg: Any) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(cfg), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


def _ensure_users_schema(conn: sqlite3.Connection) -> None:
    """
    Your live schema (confirmed):
      username TEXT PRIMARY KEY
      password_hash TEXT NOT NULL
      created_at TEXT NOT NULL
      updated_at TEXT NOT NULL
      salt TEXT NULL
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            salt TEXT
        )
        """
    )
    conn.commit()

    cols = _table_columns(conn, "users")

    # Add missing columns safely (SQLite)
    if "password_hash" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "created_at" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
    if "updated_at" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN updated_at TEXT")
    if "salt" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN salt TEXT")
    conn.commit()

    # Backfill required-ish fields
    now = _utc_now_str()
    conn.execute("UPDATE users SET created_at = COALESCE(created_at, ?) WHERE created_at IS NULL", (now,))
    conn.execute("UPDATE users SET updated_at = COALESCE(updated_at, ?) WHERE updated_at IS NULL", (now,))
    conn.execute("UPDATE users SET password_hash = COALESCE(password_hash, '') WHERE password_hash IS NULL")
    conn.commit()


# --- password handling ---
def _username_salt(username: str) -> str:
    # deterministic salt used for SHA migration path
    return hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]


def _sha256_salted(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _safe_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(str(a), str(b))


def _verify_password(username: str, password: str, stored_hash: str, salt: str) -> bool:
    stored_hash = stored_hash or ""
    salt = salt or ""

    # 1) bcrypt (your admin hash is $2b$12$...)
    if stored_hash.startswith("$2"):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except Exception:
            return False

    # 2) salted sha256 (for migrated/new users)
    if salt:
        cand = _sha256_salted(password, salt)
        return _safe_eq(cand, stored_hash)

    # 3) legacy unsalted sha256(password)
    cand_unsalted = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if _safe_eq(cand_unsalted, stored_hash):
        return True

    # 4) last resort plaintext (only if old system stored raw)
    if _safe_eq(password, stored_hash):
        return True

    # 5) try username-derived salt (if someone reset with that scheme but salt not saved)
    derived = _username_salt(username)
    cand_derived = _sha256_salted(password, derived)
    if _safe_eq(cand_derived, stored_hash):
        return True

    return False


# --- public API ---
def ensure_default_user(cfg: Any) -> None:
    """
    Creates default user if cfg.DEFAULT_USERNAME and cfg.DEFAULT_PASSWORD exist.
    NOTE: This will create a SHA256-salted user, not bcrypt.
    """
    default_username = getattr(cfg, "DEFAULT_USERNAME", None)
    default_password = getattr(cfg, "DEFAULT_PASSWORD", None)

    if not default_username or not default_password:
        return

    with _connect(cfg) as conn:
        _ensure_users_schema(conn)

        row = conn.execute(
            "SELECT username FROM users WHERE username = ?",
            (default_username,),
        ).fetchone()

        if row:
            return

        now = _utc_now_str()
        salt = _username_salt(default_username)
        pw_hash = _sha256_salted(default_password, salt)

        conn.execute(
            """
            INSERT INTO users (username, password_hash, created_at, updated_at, salt)
            VALUES (?, ?, ?, ?, ?)
            """,
            (default_username, pw_hash, now, now, salt),
        )
        conn.commit()


def verify_login(cfg: Any, username: str, password: str) -> Optional[str]:
    if not username or not password:
        return None

    with _connect(cfg) as conn:
        _ensure_users_schema(conn)
        row = conn.execute(
            "SELECT username, password_hash, COALESCE(salt,'') AS salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if not row:
        return None

    if _verify_password(
        username=row["username"],
        password=password,
        stored_hash=row["password_hash"],
        salt=row["salt"],
    ):
        return row["username"]

    return None


def change_password(cfg: Any, username: str, current_password: str, new_password: str) -> Tuple[bool, str]:
    if not username:
        return False, "Not logged in."
    if not current_password or not new_password:
        return False, "Missing password."
    if len(new_password) < 6:
        return False, "New password too short (min 6 chars)."

    # verify current password
    if not verify_login(cfg, username, current_password):
        return False, "Current password is wrong."

    # migrate/update using salted SHA256 (simple + consistent)
    salt = _username_salt(username)
    new_hash = _sha256_salted(new_password, salt)
    now = _utc_now_str()

    with _connect(cfg) as conn:
        _ensure_users_schema(conn)
        conn.execute(
            """
            UPDATE users
               SET password_hash = ?,
                   salt = ?,
                   updated_at = ?
             WHERE username = ?
            """,
            (new_hash, salt, now, username),
        )
        conn.commit()

    return True, "Password updated."