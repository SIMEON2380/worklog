import sqlite3
import pandas as pd
from datetime import datetime


def make_db(cfg):

    DB_PATH = cfg.DB_PATH
    TABLE = cfg.TABLE_NAME

    def get_conn():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    # -------------------------
    # Ensure schema
    # -------------------------
    def ensure_schema():
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(f"""
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
        """)

        conn.commit()
        conn.close()

    # -------------------------
    # Read all rows
    # -------------------------
    def read_all():
        conn = get_conn()
        df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
        conn.close()
        return df

    # -------------------------
    # Update row
    # -------------------------
    def update_row(row_id: int, diffs: dict):

        if not diffs:
            return

        conn = get_conn()
        cur = conn.cursor()

        # Get real columns
        cols = {r["name"] for r in cur.execute(f"PRAGMA table_info({TABLE})").fetchall()}

        safe = {k: v for k, v in diffs.items() if k in cols and k != "id"}

        if not safe:
            conn.close()
            return

        # update timestamp
        if "updated_at" in cols:
            safe["updated_at"] = datetime.utcnow().isoformat()

        sql = ", ".join([f"{k}=?" for k in safe.keys()])
        params = list(safe.values()) + [row_id]

        cur.execute(
            f"UPDATE {TABLE} SET {sql} WHERE id=?",
            params
        )

        conn.commit()
        conn.close()

    # -------------------------
    # DB API
    # -------------------------
    return {
        "ensure_schema": ensure_schema,
        "read_all": read_all,
        "update_row": update_row,
    }