import sqlite3
from datetime import datetime, date
from typing import Any, Dict, Optional

import pandas as pd


def make_db(cfg):
    DB_PATH = cfg.DB_PATH
    TABLE = cfg.TABLE_NAME

    def get_conn() -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")
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

    def row_exists_by_job_date(cur: sqlite3.Cursor, work_date: Any, job_id: Any) -> Optional[sqlite3.Row]:
        if not work_date or not job_id:
            return None

        return cur.execute(
            f"""
            SELECT *
            FROM {TABLE}
            WHERE work_date = ? AND job_id = ?
            LIMIT 1
            """,
            (work_date, job_id),
        ).fetchone()

    # -------------------------
    # Ensure schema
    # -------------------------
    def ensure_schema() -> None:
        conn = get_conn()
        try:
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
                    paid_date TEXT,
                    job_outcome TEXT
                )
                """
            )

            cols = get_columns(cur)

            if "add_pay" not in cols:
                cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN add_pay REAL DEFAULT 0")

            if "paid_date" not in cols:
                cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN paid_date TEXT")

            if "job_outcome" not in cols:
                cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN job_outcome TEXT")

            conn.commit()
        finally:
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
        try:
            cur = conn.cursor()

            cols = get_columns(cur)
            safe = {k: v for k, v in data.items() if k in cols and k != "id"}

            if not safe:
                return None

            work_date = safe.get("work_date")
            job_id = str(safe.get("job_id") or "").strip()

            if not work_date:
                raise ValueError("work_date is required")
            if not job_id:
                raise ValueError("job_id is required")

            safe["job_id"] = job_id

            now = datetime.utcnow().isoformat()

            if "created_at" in cols and "created_at" not in safe:
                safe["created_at"] = now
            if "updated_at" in cols and "updated_at" not in safe:
                safe["updated_at"] = now

            if "job_outcome" in cols and not safe.get("job_outcome"):
                safe["job_outcome"] = "Completed"

            if "paid_date" in cols:
                status_value = safe.get("job_status", safe.get("status"))
                if normalize_status(status_value) == "paid":
                    if not safe.get("paid_date"):
                        safe["paid_date"] = today_iso()
                else:
                    if "paid_date" not in safe:
                        safe["paid_date"] = None

            existing = row_exists_by_job_date(cur, work_date, job_id)
            if existing is not None:
                raise sqlite3.IntegrityError(
                    f"Duplicate job entry for work_date={work_date}, job_id={job_id}"
                )

            keys = list(safe.keys())
            placeholders = ", ".join(["?"] * len(keys))
            sql = f"INSERT INTO {TABLE} ({', '.join(keys)}) VALUES ({placeholders})"
            params = [safe[k] for k in keys]

            cur.execute(sql, params)
            new_id = cur.lastrowid
            conn.commit()
            return new_id
        finally:
            conn.close()

    # -------------------------
    # Update row
    # -------------------------
    def update_row(row_id: int, diffs: Dict[str, Any]) -> None:
        if not diffs:
            return

        conn = get_conn()
        try:
            cur = conn.cursor()

            cols = get_columns(cur)
            safe = {k: v for k, v in diffs.items() if k in cols and k != "id"}

            if not safe:
                return

            existing = cur.execute(
                f"SELECT * FROM {TABLE} WHERE id=?",
                (row_id,),
            ).fetchone()

            if existing is None:
                return

            new_work_date = safe.get("work_date", existing["work_date"])
            new_job_id = str(safe.get("job_id", existing["job_id"]) or "").strip()
            safe["job_id"] = new_job_id

            duplicate = cur.execute(
                f"""
                SELECT id
                FROM {TABLE}
                WHERE work_date = ? AND job_id = ? AND id <> ?
                LIMIT 1
                """,
                (new_work_date, new_job_id, row_id),
            ).fetchone()

            if duplicate is not None:
                raise sqlite3.IntegrityError(
                    f"Another row already exists for work_date={new_work_date}, job_id={new_job_id}"
                )

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
        finally:
            conn.close()

    # -------------------------
    # Delete row
    # -------------------------
    def delete_row(row_id: int) -> None:
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(f"DELETE FROM {TABLE} WHERE id=?", (row_id,))
            conn.commit()
        finally:
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