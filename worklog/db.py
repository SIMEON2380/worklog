import sqlite3
from datetime import datetime, date
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
        rows = cur.execute(f"PRAGMA table_info({TABLE})").fetchall()
        return {r["name"] for r in rows}

    def normalize_status(value: Any) -> str:
        return str(value or "").strip().lower()

    def today_iso() -> str:
        return date.today().isoformat()

    def apply_paid_date_logic(
        old_status: Any,
        new_status: Any,
        existing_paid_date: Any,
    ) -> Optional[str]:
        old_s = normalize_status(old_status)
        new_s = normalize_status(new_status)
        existing_paid_date = str(existing_paid_date).strip() if existing_paid_date else None

        if new_s == "paid" and old_s != "paid":
            return today_iso()

        if old_s == "paid" and new_s != "paid":
            return None

        if new_s == "paid":
            return existing_paid_date or today_iso()

        return existing_paid_date

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
                status TEXT,
                add_pay REAL DEFAULT 0,
                paid_date TEXT
            )
            """
        )

        cols = get_columns(cur)

        if "add_pay" not in cols:
            cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN add_pay REAL DEFAULT 0")

        if "paid_date" not in cols:
            cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN paid_date TEXT")

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
        if not data or not isinstance(data, dict):
            return None

        conn = get_conn()
        cur = conn.cursor()

        cols = get_columns(cur)
        safe = {k: v for k, v in data.items() if k in cols and k != "id"}

        now = datetime.utcnow().isoformat()

        if "created_at" in cols and "created_at" not in safe:
            safe["created_at"] = now
        if "updated_at" in cols and "updated_at" not in safe:
            safe["updated_at"] = now

        if "paid_date" in cols:
            status_value = safe.get("job_status", safe.get("status"))
            if normalize_status(status_value) == "paid":
                if not safe.get("paid_date"):
                    safe["paid_date"] = today_iso()
            else:
                if "paid_date" not in safe:
                    safe["paid_date"] = None

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
        safe = {k: v for k, v in diffs.items() if k in cols and k != "id"}

        if not safe:
            conn.close()
            return

        existing = cur.execute(
            f"SELECT * FROM {TABLE} WHERE id=?",
            (row_id,)
        ).fetchone()

        if existing is None:
            conn.close()
            return

        if "paid_date" in cols:
            old_status = existing["job_status"] if "job_status" in cols else existing["status"]
            new_status = safe.get("job_status", safe.get("status", old_status))
            existing_paid_date = existing["paid_date"] if "paid_date" in cols else None

            safe["paid_date"] = apply_paid_date_logic(
                old_status=old_status,
                new_status=new_status,
                existing_paid_date=existing_paid_date,
            )

        if "updated_at" in cols:
            safe["updated_at"] = datetime.utcnow().isoformat()

        sql = ", ".join([f"{k}=?" for k in safe.keys()])
        params = list(safe.values()) + [row_id]

        cur.execute(f"UPDATE {TABLE} SET {sql} WHERE id=?", params)

        conn.commit()
        conn.close()

    # -------------------------
    # Delete row
    # -------------------------
    def delete_row(row_id: int) -> None:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {TABLE} WHERE id=?", (row_id,))
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
        "delete_row": delete_row,
    }