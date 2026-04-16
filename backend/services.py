from typing import Optional, List
from fastapi import HTTPException
from .db import get_connection


def list_jobs(
    work_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    conn = get_connection()
    cur = conn.cursor()

    query = "SELECT * FROM jobs"
    params = []

    # Optional filter by date
    if work_date:
        query += " WHERE work_date = ?"
        params.append(work_date)

    query += " ORDER BY work_date DESC, id DESC"

    # Optional limit (only if provided)
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    cur.execute(query, params)

    columns = [col[0] for col in cur.description]
    rows = cur.fetchall()

    conn.close()

    return [dict(zip(columns, row)) for row in rows]