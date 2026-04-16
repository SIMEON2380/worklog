from typing import Optional, List
from fastapi import HTTPException
from .db import get_connection


def list_jobs(
    work_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    conn = get_connection()
    cur = conn.cursor()

    # ✅ FIXED TABLE NAME
    query = "SELECT * FROM entries"
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

    try:
        cur.execute(query, params)

        columns = [col[0] for col in cur.description]
        rows = cur.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()