import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Iterable, List, Optional, Tuple

import pandas as pd

from .config import Config


def _db_path(cfg: Config) -> str:
    os.makedirs(cfg.DB_DIR, exist_ok=True)
    return os.path.join(cfg.DB_DIR, cfg.DB_FILE)


@contextmanager
def connect(cfg: Config):
    conn = sqlite3.connect(_db_path(cfg), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _table_columns(conn: sqlite3.Connection, table: str) -> Dict[str, str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = {}
    for r in cur.fetchall():
        cols[r["name"]] = r["type"]
    return cols


def init_db(cfg: Config) -> None:
    """
    Creates tables if missing and adds missing columns safely.
    This is migration-safe: it never drops existing data.
    """
    with connect(cfg) as conn:
        # Users table
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {cfg.USERS_TABLE} (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        # Work logs table (safe default schema)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {cfg.TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_date TEXT NOT NULL,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                reference TEXT DEFAULT '',
                start_time TEXT DEFAULT '',
                end_time TEXT DEFAULT '',
                waiting_hours REAL DEFAULT 0,
                pay REAL DEFAULT 0,
                expense_type TEXT DEFAULT '',
                expense_amount REAL DEFAULT 0,
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        # Add missing columns safely if older schema exists
        existing = _table_columns(conn, cfg.TABLE_NAME)
        desired = {
            "id": "INTEGER",
            "job_date": "TEXT",
            "job_type": "TEXT",
            "status": "TEXT",
            "reference": "TEXT",
            "start_time": "TEXT",
            "end_time": "TEXT",
            "waiting_hours": "REAL",
            "pay": "REAL",
            "expense_type": "TEXT",
            "expense_amount": "REAL",
            "notes": "TEXT",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        }

        for col, col_type in desired.items():
            if col not in existing:
                # SQLite allows ADD COLUMN; no DROP.
                conn.execute(f"ALTER TABLE {cfg.TABLE_NAME} ADD COLUMN {col} {col_type}")
        # Ensure updated_at/created_at exist for older rows
        now = datetime.utcnow().isoformat(timespec="seconds")
        if "created_at" in desired:
            conn.execute(
                f"UPDATE {cfg.TABLE_NAME} SET created_at = COALESCE(created_at, ?) WHERE created_at IS NULL OR created_at = ''",
                (now,),
            )
        if "updated_at" in desired:
            conn.execute(
                f"UPDATE {cfg.TABLE_NAME} SET updated_at = COALESCE(updated_at, ?) WHERE updated_at IS NULL OR updated_at = ''",
                (now,),
            )


def fetch_jobs(
    cfg: Config,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    search: Optional[str] = None,
) -> pd.DataFrame:
    where = []
    params: List[Any] = []

    if date_from:
        where.append("job_date >= ?")
        params.append(date_from)
    if date_to:
        where.append("job_date <= ?")
        params.append(date_to)
    if status and status != "All":
        where.append("status = ?")
        params.append(status)
    if job_type and job_type != "All":
        where.append("job_type = ?")
        params.append(job_type)
    if search:
        where.append("(notes LIKE ? OR reference LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT
            id, job_date, job_type, status, reference, start_time, end_time,
            waiting_hours, pay, expense_type, expense_amount, notes,
            created_at, updated_at
        FROM {cfg.TABLE_NAME}
        {where_sql}
        ORDER BY job_date DESC, id DESC
    """

    with connect(cfg) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    # Friendly types
    if not df.empty:
        df["waiting_hours"] = pd.to_numeric(df["waiting_hours"], errors="coerce").fillna(0.0)
        df["pay"] = pd.to_numeric(df["pay"], errors="coerce").fillna(0.0)
        df["expense_amount"] = pd.to_numeric(df["expense_amount"], errors="coerce").fillna(0.0)
    return df


def update_job_status(cfg: Config, job_id: int, new_status: str) -> None:
    from datetime import datetime

    now = datetime.utcnow().isoformat(timespec="seconds")
    with connect(cfg) as conn:
        conn.execute(
            f"UPDATE {cfg.TABLE_NAME} SET status = ?, updated_at = ? WHERE id = ?",
            (new_status, now, int(job_id)),
        )


def update_job_fields(cfg: Config, job_id: int, fields: Dict[str, Any]) -> None:
    """
    Updates editable fields safely. Only updates known columns.
    """
    from datetime import datetime

    allowed = {
        "job_date", "job_type", "status", "reference",
        "start_time", "end_time", "waiting_hours", "pay",
        "expense_type", "expense_amount", "notes",
    }

    clean = {k: v for k, v in fields.items() if k in allowed}
    if not clean:
        return

    clean["updated_at"] = datetime.utcnow().isoformat(timespec="seconds")

    sets = ", ".join([f"{k} = ?" for k in clean.keys()])
    params = list(clean.values()) + [int(job_id)]

    with connect(cfg) as conn:
        conn.execute(
            f"UPDATE {cfg.TABLE_NAME} SET {sets} WHERE id = ?",
            params,
        )


def insert_job(cfg: Config, row: Dict[str, Any]) -> None:
    from datetime import datetime

    now = datetime.utcnow().isoformat(timespec="seconds")

    data = {
        "job_date": row.get("job_date", ""),
        "job_type": row.get("job_type", ""),
        "status": row.get("status", "Start"),
        "reference": row.get("reference", ""),
        "start_time": row.get("start_time", ""),
        "end_time": row.get("end_time", ""),
        "waiting_hours": float(row.get("waiting_hours") or 0),
        "pay": float(row.get("pay") or 0),
        "expense_type": row.get("expense_type", ""),
        "expense_amount": float(row.get("expense_amount") or 0),
        "notes": row.get("notes", ""),
        "created_at": now,
        "updated_at": now,
    }

    cols = ", ".join(data.keys())
    qs = ", ".join(["?"] * len(data))
    with connect(cfg) as conn:
        conn.execute(
            f"INSERT INTO {cfg.TABLE_NAME} ({cols}) VALUES ({qs})",
            list(data.values()),
        )