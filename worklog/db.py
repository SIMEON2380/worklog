# worklog/db.py
import os
import sqlite3
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

TABLE_NAME = "work_logs"


def get_db_path() -> str:
    db_dir = os.environ.get("WORKLOG_DB_DIR", "/var/lib/worklog")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "worklog.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def _existing_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({TABLE_NAME});").fetchall()
    return {r["name"] for r in rows}


def init_db() -> None:
    """
    Your table already exists. We ONLY do safe schema upgrades here:
    add columns your new UI expects (postcode/customer_name/site_address),
    and optionally updated_at.
    """
    conn = get_conn()
    try:
        cols = _existing_columns(conn)

        # Add columns the UI expects but your old schema doesn't have.
        # These are safe ADD COLUMN operations (no data loss).
        upgrades: List[Tuple[str, str]] = [
            ("postcode", "TEXT"),
            ("customer_name", "TEXT"),
            ("site_address", "TEXT"),
            ("updated_at", "TEXT"),
        ]

        for col, ddl in upgrades:
            if col not in cols:
                conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {col} {ddl};")

        conn.commit()
    finally:
        conn.close()


# --- Internal helpers ---------------------------------------------------------

def _latest_row_id_for_job(conn: sqlite3.Connection, job_id: str) -> Optional[int]:
    row = conn.execute(
        f"""
        SELECT id
        FROM {TABLE_NAME}
        WHERE job_id = ?
        ORDER BY
          CASE WHEN work_date IS NULL OR work_date = '' THEN 0 ELSE 1 END DESC,
          work_date DESC,
          id DESC
        LIMIT 1;
        """,
        (job_id,),
    ).fetchone()
    return int(row["id"]) if row else None


def _to_ui_row(db_row: sqlite3.Row) -> Dict[str, Any]:
    """
    Convert DB row (old schema) to keys expected by app.py.
    """
    return {
        "job_number": db_row["job_id"],
        "job_type": db_row["category"],
        "status": db_row["job_status"],
        "vehicle_description": db_row["vehicle_description"],
        "postcode": db_row["postcode"],
        "expense_type": db_row["job_expenses"],
        "customer_name": db_row["customer_name"],
        "site_address": db_row["site_address"],
        "notes": db_row["comments"],
        # keep some useful originals if you ever want to show them
        "id": db_row["id"],
        "work_date": db_row["work_date"],
        "hours": db_row["hours"],
        "amount": db_row["amount"],
        "created_at": db_row["created_at"],
        "updated_at": db_row["updated_at"],
    }


# --- Public API used by app.py -----------------------------------------------

def list_jobs(search: str = "", limit: int = 500) -> pd.DataFrame:
    """
    The old table can have multiple rows per job_id (because unique index is work_date + job_id).
    For the UI, we show ONE row per job_id: the latest one.
    """
    conn = get_conn()
    search = (search or "").strip()

    try:
        # Use a window function to pick the latest row per job_id.
        # (SQLite supports this in modern versions.)
        base = f"""
        WITH ranked AS (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY job_id
              ORDER BY
                CASE WHEN work_date IS NULL OR work_date = '' THEN 0 ELSE 1 END DESC,
                work_date DESC,
                id DESC
            ) AS rn
          FROM {TABLE_NAME}
        )
        SELECT
          id,
          work_date,
          hours,
          amount,
          created_at,
          updated_at,

          job_id            AS job_number,
          category          AS job_type,
          job_status        AS status,
          vehicle_description,
          postcode,
          job_expenses      AS expense_type,
          customer_name,
          site_address,
          comments          AS notes

        FROM ranked
        WHERE rn = 1
        """

        params: Tuple[Any, ...]
        if search:
            like = f"%{search}%"
            sql = base + """
              AND (
                job_id LIKE ?
                OR COALESCE(postcode,'') LIKE ?
                OR COALESCE(vehicle_description,'') LIKE ?
                OR COALESCE(customer_name,'') LIKE ?
              )
              ORDER BY job_number DESC
              LIMIT ?
            """
            params = (like, like, like, like, int(limit))
        else:
            sql = base + """
              ORDER BY job_number DESC
              LIMIT ?
            """
            params = (int(limit),)

        df = pd.read_sql_query(sql, conn, params=params)
        return df
    finally:
        conn.close()


def get_job_by_number(job_number: str) -> Optional[Dict[str, Any]]:
    """
    In your DB: job_number == job_id.
    Returns the latest row for that job_id mapped to UI field names.
    """
    conn = get_conn()
    try:
        row = conn.execute(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE job_id = ?
            ORDER BY
              CASE WHEN work_date IS NULL OR work_date = '' THEN 0 ELSE 1 END DESC,
              work_date DESC,
              id DESC
            LIMIT 1;
            """,
            (job_number,),
        ).fetchone()

        return _to_ui_row(row) if row else None
    finally:
        conn.close()


def upsert_job(record: Dict[str, Any]) -> None:
    """
    Your UI doesn't supply work_date, but your old schema is date-based.
    So we treat "upsert" as:
      - update latest row for that job_id if it exists
      - else insert a new row with work_date = today and blank numeric fields
    """
    job_id = (record or {}).get("job_number")
    if not job_id:
        raise ValueError("record must include job_number")

    conn = get_conn()
    try:
        existing_id = _latest_row_id_for_job(conn, job_id)

        if existing_id is not None:
            conn.execute(
                f"""
                UPDATE {TABLE_NAME}
                SET
                  category = ?,
                  job_status = ?,
                  vehicle_description = ?,
                  postcode = ?,
                  job_expenses = ?,
                  customer_name = ?,
                  site_address = ?,
                  comments = ?,
                  updated_at = datetime('now')
                WHERE id = ?;
                """,
                (
                    record.get("job_type"),
                    record.get("status"),
                    record.get("vehicle_description"),
                    record.get("postcode"),
                    record.get("expense_type"),
                    record.get("customer_name"),
                    record.get("site_address"),
                    record.get("notes"),
                    existing_id,
                ),
            )
        else:
            conn.execute(
                f"""
                INSERT INTO {TABLE_NAME} (
                  work_date, description, hours, amount,
                  job_id, category,
                  job_status,
                  vehicle_description, postcode,
                  job_expenses,
                  customer_name, site_address,
                  comments,
                  created_at, updated_at
                )
                VALUES (
                  date('now'), '', 0, 0,
                  ?, ?,
                  ?,
                  ?, ?,
                  ?,
                  ?, ?,
                  ?,
                  datetime('now'), datetime('now')
                );
                """,
                (
                    job_id,
                    record.get("job_type"),
                    record.get("status"),
                    record.get("vehicle_description"),
                    record.get("postcode"),
                    record.get("expense_type"),
                    record.get("customer_name"),
                    record.get("site_address"),
                    record.get("notes"),
                ),
            )

        conn.commit()
    finally:
        conn.close()


def update_job(
    *,
    job_number: str,
    job_type: str = "",
    status: str = "",
    vehicle_description: str = "",
    postcode: str = "",
    expense_type: str = "",
    customer_name: str = "",
    site_address: str = "",
    notes: str = "",
) -> bool:
    """
    Updates the latest row for that job_id.
    Returns True if updated, False if job doesn't exist.
    """
    conn = get_conn()
    try:
        row_id = _latest_row_id_for_job(conn, job_number)
        if row_id is None:
            return False

        cur = conn.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET
              category = ?,
              job_status = ?,
              vehicle_description = ?,
              postcode = ?,
              job_expenses = ?,
              customer_name = ?,
              site_address = ?,
              comments = ?,
              updated_at = datetime('now')
            WHERE id = ?;
            """,
            (
                job_type,
                status,
                vehicle_description,
                postcode,
                expense_type,
                customer_name,
                site_address,
                notes,
                row_id,
            ),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_job(job_number: str) -> bool:
    """
    Deletes ALL rows for that job_id (because the old schema can have multiple days per job).
    """
    conn = get_conn()
    try:
        cur = conn.execute(
            f"DELETE FROM {TABLE_NAME} WHERE job_id = ?;",
            (job_number,),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# Backwards-compat alias (if you still import these somewhere)
def get_job_by_number_or_none(job_number: str) -> Optional[Dict[str, Any]]:
    return get_job_by_number(job_number)


def update_job_by_number(*args, **kwargs) -> bool:
    return update_job(*args, **kwargs)