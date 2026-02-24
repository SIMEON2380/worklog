# worklog/db.py
import os
import sqlite3
from typing import Any, Dict, Optional, Tuple

import pandas as pd

TABLE_NAME = "work_logs"


def get_db_path() -> str:
    db_dir = os.environ.get("WORKLOG_DB_DIR", "/var/lib/worklog")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "worklog.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db() -> None:
    # Your table already exists. We do NOT alter schema.
    return


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


def list_jobs(search: str = "", limit: int = 500) -> pd.DataFrame:
    """
    Show one row per job_id (latest entry), but keep ORIGINAL column meaning.
    We alias to UI-friendly names ONLY if your UI expects them.
    """
    conn = get_conn()
    search = (search or "").strip()

    try:
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
          description,
          hours,
          amount,

          job_id               AS job_number,
          category             AS job_type,
          job_status           AS status,
          vehicle_description,
          vehicle_reg,
          collection_from,
          delivery_to,
          job_expenses,
          expenses_amount,
          auth_code,
          comments

        FROM ranked
        WHERE rn = 1
        """

        params: Tuple[Any, ...]
        if search:
            like = f"%{search}%"
            sql = base + """
              AND (
                job_id LIKE ?
                OR COALESCE(vehicle_description,'') LIKE ?
                OR COALESCE(vehicle_reg,'') LIKE ?
                OR COALESCE(collection_from,'') LIKE ?
                OR COALESCE(delivery_to,'') LIKE ?
                OR COALESCE(comments,'') LIKE ?
              )
              ORDER BY job_number DESC
              LIMIT ?
            """
            params = (like, like, like, like, like, like, int(limit))
        else:
            sql = base + """
              ORDER BY job_number DESC
              LIMIT ?
            """
            params = (int(limit),)

        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def get_job_by_number(job_number: str) -> Optional[Dict[str, Any]]:
    """
    Returns latest row for job_id, mapped to the names your UI uses.
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

        if not row:
            return None

        # Map to what the UI expects
        return {
            "job_number": row["job_id"],
            "job_type": row["category"],
            "status": row["job_status"],
            "vehicle_description": row["vehicle_description"],
            "vehicle_reg": row["vehicle_reg"],
            "collection_from": row["collection_from"],
            "delivery_to": row["delivery_to"],
            "job_expenses": row["job_expenses"],
            "expenses_amount": row["expenses_amount"],
            "auth_code": row["auth_code"],
            "comments": row["comments"],
        }
    finally:
        conn.close()


def upsert_job(record: Dict[str, Any]) -> None:
    """
    Update latest row for job_id if it exists; otherwise insert a new row for today.
    Only touches ORIGINAL columns.
    """
    job_id = (record or {}).get("job_number")
    if not job_id:
        raise ValueError("record must include job_number")

    conn = get_conn()
    try:
        row_id = _latest_row_id_for_job(conn, job_id)

        if row_id is not None:
            conn.execute(
                f"""
                UPDATE {TABLE_NAME}
                SET
                  category = ?,
                  job_status = ?,
                  vehicle_description = ?,
                  vehicle_reg = ?,
                  collection_from = ?,
                  delivery_to = ?,
                  job_expenses = ?,
                  expenses_amount = ?,
                  auth_code = ?,
                  comments = ?
                WHERE id = ?;
                """,
                (
                    record.get("job_type"),
                    record.get("status"),
                    record.get("vehicle_description"),
                    record.get("vehicle_reg"),
                    record.get("collection_from"),
                    record.get("delivery_to"),
                    record.get("job_expenses"),
                    record.get("expenses_amount"),
                    record.get("auth_code"),
                    record.get("comments"),
                    row_id,
                ),
            )
        else:
            conn.execute(
                f"""
                INSERT INTO {TABLE_NAME} (
                  work_date, description, hours, amount,
                  job_id, category, job_status,
                  vehicle_description, vehicle_reg,
                  collection_from, delivery_to,
                  job_expenses, expenses_amount,
                  auth_code, comments,
                  created_at
                )
                VALUES (
                  date('now'), '', 0, 0,
                  ?, ?, ?,
                  ?, ?,
                  ?, ?,
                  ?, ?,
                  ?, ?,
                  datetime('now')
                );
                """,
                (
                    job_id,
                    record.get("job_type"),
                    record.get("status"),
                    record.get("vehicle_description"),
                    record.get("vehicle_reg"),
                    record.get("collection_from"),
                    record.get("delivery_to"),
                    record.get("job_expenses"),
                    record.get("expenses_amount"),
                    record.get("auth_code"),
                    record.get("comments"),
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
    vehicle_reg: str = "",
    collection_from: str = "",
    delivery_to: str = "",
    job_expenses: str = "",
    expenses_amount: float = 0.0,
    auth_code: str = "",
    comments: str = "",
) -> bool:
    """
    Update latest row for job_id. Only ORIGINAL columns.
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
              vehicle_reg = ?,
              collection_from = ?,
              delivery_to = ?,
              job_expenses = ?,
              expenses_amount = ?,
              auth_code = ?,
              comments = ?
            WHERE id = ?;
            """,
            (
                job_type,
                status,
                vehicle_description,
                vehicle_reg,
                collection_from,
                delivery_to,
                job_expenses,
                expenses_amount,
                auth_code,
                comments,
                row_id,
            ),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_job(job_number: str) -> bool:
    """
    Deletes all rows for job_id (matches your old data model).
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