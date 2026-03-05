import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


def make_db(cfg):
    DB_PATH = cfg.DB_PATH
    TABLE = cfg.TABLE_NAME

    def get_conn() -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def get_columns(cur: sqlite3.Cursor) -> set[str]:
        # PRAGMA table_info returns rows with keys: cid, name, type, notnull, dflt_value, pk
        rows = cur.execute(f"PRAGMA table_info({TABLE})").fetchall()
        return {r["name"] for r in rows}

    # -------------------------
    # Ensure schema
    # -------------------------
    def ensure_schema() -> None:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                work_date TEXT,
                description TEXT,
                hours REAL,
                amount REAL,
                job_id TEXT,
                category TEXT,
                job_status TEXT,
                waiting_time TEXT,
                waiting_hours REAL,
                waiting_amount REAL,
                created_at TEXT,
                vehicle_description TEXT,
                vehicle_reg TEXT,
                collection_from TEXT,
                delivery_to TEXT,
                job_expenses TEXT,
                expenses_amount REAL,
                auth_code TEXT,
                comments TEXT,
                postcode TEXT,
                customer_name TEXT,
                site_address TEXT,
                updated_at TEXT,
                status TEXT
            )
            """
        )

        conn.commit()
        conn.close()

    # -------------------------
    # Read all rows
    # -------------------------
    def read_all() -> pd.DataFrame:
        conn = get_conn()
        try:
            df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
            return df
        finally:
            conn.close()

    # -------------------------
    # Insert row
    # -------------------------
    def insert_row(data: Dict[str, Any]) -> Optional[int]:
        """
        Inserts one row into TABLE.
        - Filters incoming data to real DB columns
        - Adds created_at/updated_at if present and missing
        Returns the new row id (or None if nothing inserted).
        """
        if not data or not isinstance(data, dict):
            return None

        conn = get_conn()
        cur = conn.cursor()

        cols = get_columns(cur)

        # Only allow real columns (never allow setting id manually)
        safe = {k: v for k, v in data.items() if k in cols and k != "id"}

        # Auto timestamps if supported
        now = datetime.utcnow().isoformat()
        if "created_at" in cols and "created_at" not in safe:
            safe["created_at"] = now
        if "updated_at" in cols and "updated_at" not in safe:
            safe["updated_at"] = now

        if not safe:
            conn.close()
            return None

        keys = list(safe.keys())
        placeholders = ", ".join(["?"] * len(keys))
        sql = f"INSERT INTO {TABLE} ({', '.join(keys)}) VALUES ({placeholders})"
        params = [safe[k] for k in keys]

        cur.execute(sql, params)
        new_id = cur.lastrowid

        conn.commit()
        conn.close()
        return new_id

    # -------------------------
    # Update row
    # -------------------------
    def update_row(row_id: int, diffs: Dict[str, Any]) -> None:
        if not diffs:
            return

        conn = get_conn()
        cur = conn.cursor()

        cols = get_columns(cur)

        # Only allow real columns (never allow changing id)
        safe = {k: v for k, v in diffs.items() if k in cols and k != "id"}

        if not safe:
            conn.close()
            return

        # update timestamp
        if "updated_at" in cols:
            safe["updated_at"] = datetime.utcnow().isoformat()

        sql = ", ".join([f"{k}=?" for k in safe.keys()])
        params = list(safe.values()) + [row_id]

        cur.execute(f"UPDATE {TABLE} SET {sql} WHERE id=?", params)

        conn.commit()
        conn.close()

    # -------------------------
    # DB API
    # -------------------------
    return {
        "ensure_schema": ensure_schema,
        "read_all": read_all,
        "insert_row": insert_row,
        "update_row": update_row,
    }