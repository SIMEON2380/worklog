from typing import Optional, List
from fastapi import HTTPException
from .db import get_connection


def list_jobs(
    work_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    conn = get_connection()
    cur = conn.cursor()

    # ✅ NORMALIZED SCHEMA (entries → jobs-style output)
    query = """
        SELECT
            id,
            work_date,
            job_number AS job_id,
            job_type AS category,
            job_status,
            job_amount AS amount,
            NULL AS waiting_time,
            0 AS waiting_hours,
            0 AS waiting_amount,
            vehicle_description,
            vehicle_reg,
            from_loc AS collection_from,
            to_loc AS delivery_to,
            expenses AS job_expenses,
            expense_amount AS expenses_amount,
            auth_code,
            comments,
            0 AS add_pay,
            NULL AS paid_date,
            NULL AS job_outcome,
            created_at,
            updated_at
        FROM entries
    """
    params = []

    # Optional filter by date
    if work_date:
        query += " WHERE work_date = ?"
        params.append(work_date)

    query += " ORDER BY work_date DESC, id DESC"

    # Optional limit
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    try:
        cur.execute(query, params)

        columns = [col[0] for col in cur.description]
        rows = cur.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()