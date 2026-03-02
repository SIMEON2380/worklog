import sqlite3
import hashlib
import hmac
from typing import Optional, Tuple, Any


def _get_cfg(cfg_or_db: Any):
    if isinstance(cfg_or_db, dict):
        if "cfg" in cfg_or_db:
            return cfg_or_db["cfg"]
        if "__cfg__" in cfg_or_db:
            return cfg_or_db["__cfg__"]
    return cfg_or_db


def _db_path(cfg_or_db: Any) -> str:
    cfg = _get_cfg(cfg_or_db)
    if not hasattr(cfg, "DB_PATH"):
        raise AttributeError("Config must have DB_PATH.")
    return cfg.DB_PATH


def _connect(cfg_or_db: Any):
    conn = sqlite3.connect(_db_path(cfg_or_db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _verify_password(password: str, salt: str, stored_hash: str) -> bool:
    cand = _hash_password(password, salt)
    return hmac.compare_digest(cand, stored_hash)


def _get_user_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(users)").fetchall()
    return {r["name"] for r in rows}


def _ensure_users_schema(conn: sqlite3.Connection):
    """
    Makes users table compatible with:
      username, salt, password_hash
    Also supports legacy:
      username, password
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL
        )
        """
    )
    conn.commit()

    cols = _get_user_columns(conn)

    # Add missing columns
    if "salt" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN salt TEXT")
        conn.commit()
        cols.add("salt")

    if "password_hash" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        conn.commit()
        cols.add("password_hash")

    # If legacy 'password' exists, copy into password_hash where missing
    if "password" in cols:
        conn.execute(
            """
            UPDATE users
               SET password_hash = COALESCE(password_hash, password)
             WHERE password_hash IS NULL
            """
        )
        conn.commit()

    # Ensure salt not null (use empty string default for old rows)
    conn.execute("UPDATE users SET salt = COALESCE(salt, '') WHERE salt IS NULL")
    conn.commit()


def ensure_default_user(cfg_or_db: Any):
    cfg = _get_cfg(cfg_or_db)
    with _connect(cfg) as conn:
        _ensure_users_schema(conn)

        default_username = getattr(cfg, "DEFAULT_USERNAME", None)
        default_password = getattr(cfg, "DEFAULT_PASSWORD", None)

        if not default_username or not default_password:
            return

        row = conn.execute(
            "SELECT 1 FROM users WHERE username = ?",
            (default_username,),
        ).fetchone()

        if row:
            return

        # Create salted hash for default user
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
        _ensure_users_schema(conn)
        cols = _get_user_columns(conn)

        # Prefer new schema
        if "salt" in cols and "password_hash" in cols:
            row = conn.execute(
                "SELECT username, salt, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()

            if not row:
                return None

            salt = row["salt"] or ""
            stored = row["password_hash"]

            # If no stored hash yet, fail
            if not stored:
                return None

            # If salt empty, treat as legacy unsalted hash storage
            if salt == "":
                # legacy: password_hash is either plain or already hashed
                # try compare against SHA256(password) first, then raw compare
                sha = hashlib.sha256(password.encode("utf-8")).hexdigest()
                if hmac.compare_digest(stored, sha) or hmac.compare_digest(stored, password):
                    return row["username"]
                return None

            # normal salted verify
            if _verify_password(password, salt, stored):
                return row["username"]

            return None

        # Ultra-legacy fallback: username + password column only
        if "password" in cols:
            row = conn.execute(
                "SELECT username, password FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if not row:
                return None

            stored = row["password"]
            sha = hashlib.sha256(password.encode("utf-8")).hexdigest()
            if hmac.compare_digest(stored, sha) or hmac.compare_digest(stored, password):
                return row["username"]
            return None

    return None


def change_password(cfg_or_db: Any, username: str, current_password: str, new_password: str) -> Tuple[bool, str]:
    if not username:
        return False, "Not logged in."
    if not current_password or not new_password:
        return False, "Missing password."
    if len(new_password) < 6:
        return False, "New password too short (min 6 chars)."

    with _connect(cfg_or_db) as conn:
        _ensure_users_schema(conn)

        # Verify current login first
        ok_user = verify_login(cfg_or_db, username, current_password)
        if not ok_user:
            return False, "Current password is wrong."

        # Set new salted hash
        salt = hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]
        pw_hash = _hash_password(new_password, salt)

        conn.execute(
            "UPDATE users SET salt = ?, password_hash = ? WHERE username = ?",
            (salt, pw_hash, username),
        )
        conn.commit()

    return True, "Password updated."