import os
import sqlite3
from typing import Optional, Dict, Any
import pandas as pd

TABLE_NAME = "work_logs"


def get_db_path() -> str:
    db_dir = os.environ.get("WORKLOG_DB_DIR", "/var/lib/worklog")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "worklog.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            job_number TEXT PRIMARY KEY,
            job_type TEXT,
            status TEXT,
            vehicle_description TEXT,
            postcode TEXT,
            expense_type TEXT,
            customer_name TEXT,
            site_address TEXT,
            notes TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def upsert_job(record: Dict[str, Any]) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO {TABLE_NAME} (
            job_number, job_type, status, vehicle_description, postcode,
            expense_type, customer_name, site_address, notes, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(job_number) DO UPDATE SET
            job_type=excluded.job_type,
            status=excluded.status,
            vehicle_description=excluded.vehicle_description,
            postcode=excluded.postcode,
            expense_type=excluded.expense_type,
            customer_name=excluded.customer_name,
            site_address=excluded.site_address,
            notes=excluded.notes,
            updated_at=datetime('now')
    """, (
        record.get("job_number"),
        record.get("job_type"),
        record.get("status"),
        record.get("vehicle_description"),
        record.get("postcode"),
        record.get("expense_type"),
        record.get("customer_name"),
        record.get("site_address"),
        record.get("notes"),
    ))
    conn.commit()
    conn.close()


def list_jobs(search: str = "", limit: int = 500) -> pd.DataFrame:
    conn = get_conn()
    search = (search or "").strip()

    if search:
        like = f"%{search}%"
        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE job_number LIKE ?
               OR postcode LIKE ?
               OR vehicle_description LIKE ?
               OR customer_name LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            conn,
            params=(like, like, like, like, limit),
        )
    else:
        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )

    conn.close()
    return df


def get_job_by_number(job_number: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {TABLE_NAME} WHERE job_number = ?", (job_number,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None
    return dict(row)


def update_job(
    job_number: str,
    job_type: str,
    status: str,
    vehicle_description: str,
    postcode: str,
    expense_type: str,
    customer_name: str,
    site_address: str,
    notes: str,
) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"""
        UPDATE {TABLE_NAME}
        SET
            job_type = ?,
            status = ?,
            vehicle_description = ?,
            postcode = ?,
            expense_type = ?,
            customer_name = ?,
            site_address = ?,
            notes = ?,
            updated_at = datetime('now')
        WHERE job_number = ?
    """, (
        job_type,
        status,
        vehicle_description,
        postcode,
        expense_type,
        customer_name,
        site_address,
        notes,
        job_number,
    ))
    conn.commit()
    conn.close()


def delete_job(job_number: str) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {TABLE_NAME} WHERE job_number = ?", (job_number,))
    deleted = cur.rowcount > 0
    conn.commit()
    conn.close()
    return deleted