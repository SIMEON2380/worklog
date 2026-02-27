import hmac
from datetime import datetime
from typing import Optional

import bcrypt

from .config import Config
from .db import connect


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def ensure_default_user(cfg: Config) -> None:
    """
    If no users exist, create a default admin/admin123.
    You should change it immediately using Change Password.
    """
    with connect(cfg) as conn:
        cur = conn.execute(f"SELECT COUNT(*) AS c FROM {cfg.USERS_TABLE}")
        c = cur.fetchone()["c"]
        if c and int(c) > 0:
            return

        default_user = "admin"
        default_pass = "admin123"
        pw_hash = bcrypt.hashpw(default_pass.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        conn.execute(
            f"""
            INSERT INTO {cfg.USERS_TABLE} (username, password_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (default_user, pw_hash, _now(), _now()),
        )


def verify_login(cfg: Config, username: str, password: str) -> bool:
    with connect(cfg) as conn:
        cur = conn.execute(
            f"SELECT password_hash FROM {cfg.USERS_TABLE} WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return False

        stored = row["password_hash"].encode("utf-8")
        ok = bcrypt.checkpw(password.encode("utf-8"), stored)
        return bool(ok)


def change_password(cfg: Config, username: str, old_password: str, new_password: str) -> str:
    if len(new_password) < cfg.PASSWORD_MIN_LEN:
        return f"Password must be at least {cfg.PASSWORD_MIN_LEN} characters."

    with connect(cfg) as conn:
        cur = conn.execute(
            f"SELECT password_hash FROM {cfg.USERS_TABLE} WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return "User not found."

        stored = row["password_hash"].encode("utf-8")
        if not bcrypt.checkpw(old_password.encode("utf-8"), stored):
            return "Old password is wrong."

        new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        conn.execute(
            f"UPDATE {cfg.USERS_TABLE} SET password_hash = ?, updated_at = ? WHERE username = ?",
            (new_hash, _now(), username),
        )

    return "Password updated."