from typing import Optional, List, Any

from fastapi import HTTPException

from .db import get_connection


NORMALIZED_SELECT = """
    SELECT
        id,
        work_date,
        job_id,
        category,
        COALESCE(job_status, status) AS job_status,
        amount,
        waiting_time,
        COALESCE(waiting_hours, 0) AS waiting_hours,
        COALESCE(waiting_amount, 0) AS waiting_amount,
        vehicle_description,
        vehicle_reg,
        collection_from,
        delivery_to,
        job_expenses,
        COALESCE(expenses_amount, 0) AS expenses_amount,
        auth_code,
        comments,
        COALESCE(add_pay, 0) AS add_pay,
        paid_date,
        job_outcome,
        created_at,
        updated_at
    FROM work_logs
"""


def _rows_to_dicts(cur, rows) -> List[dict]:
    columns = [col[0] for col in cur.description]
    return [dict(zip(columns, row)) for row in rows]


def _run_select(query: str, params: Optional[List[Any]] = None, one: bool = False):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(query, params or [])

        if one:
            row = cur.fetchone()
            if not row:
                return None
            columns = [col[0] for col in cur.description]
            return dict(zip(columns, row))

        rows = cur.fetchall()
        return _rows_to_dicts(cur, rows)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()


def list_jobs(
    work_date: Optional[str] = None,
    limit: Optional[int] = None,
    job_status: Optional[str] = None,
    job_outcome: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    query = NORMALIZED_SELECT
    where_clauses = []
    params: List[Any] = []

    if work_date:
        where_clauses.append("work_date = ?")
        params.append(work_date)

    if job_status:
        where_clauses.append("COALESCE(job_status, status) = ?")
        params.append(job_status)

    if job_outcome:
        where_clauses.append("job_outcome = ?")
        params.append(job_outcome)

    if category:
        where_clauses.append("category = ?")
        params.append(category)

    if start_date:
        where_clauses.append("work_date >= ?")
        params.append(start_date)

    if end_date:
        where_clauses.append("work_date <= ?")
        params.append(end_date)

    if search:
        where_clauses.append(
            """(
                CAST(job_id AS TEXT) LIKE ?
                OR COALESCE(vehicle_reg, '') LIKE ?
                OR COALESCE(collection_from, '') LIKE ?
                OR COALESCE(delivery_to, '') LIKE ?
                OR COALESCE(comments, '') LIKE ?
                OR COALESCE(auth_code, '') LIKE ?
            )"""
        )
        term = f"%{search}%"
        params.extend([term, term, term, term, term, term])

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY work_date DESC, id DESC"

    if limit:
        query += " LIMIT ?"
        params.append(limit)
        data = _run_select(query, params, one=False)
        return {"data": data}

    offset = (page - 1) * page_size
    query += " LIMIT ? OFFSET ?"
    params.extend([page_size, offset])

    data = _run_select(query, params, one=False)
    return {
        "data": data,
        "page": page,
        "page_size": page_size,
    }


def get_job_by_id(job_id: str) -> dict:
    query = (
        NORMALIZED_SELECT
        + " WHERE CAST(job_id AS TEXT) = ? ORDER BY work_date DESC, id DESC LIMIT 1"
    )
    result = _run_select(query, [str(job_id)], one=True)

    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    return result


def get_job_by_row_id(row_id: int) -> dict:
    query = NORMALIZED_SELECT + " WHERE id = ? LIMIT 1"
    result = _run_select(query, [row_id], one=True)

    if not result:
        raise HTTPException(status_code=404, detail="Job row not found")

    return result


def create_job_record(job) -> dict:
    data = job.model_dump(exclude_unset=True)

    field_map = {
        "work_date": "work_date",
        "job_id": "job_id",
        "category": "category",
        "vehicle_description": "vehicle_description",
        "vehicle_reg": "vehicle_reg",
        "collection_from": "collection_from",
        "delivery_to": "delivery_to",
        "amount": "amount",
        "job_expenses": "job_expenses",
        "expenses_amount": "expenses_amount",
        "comments": "comments",
        "auth_code": "auth_code",
        "job_status": "job_status",
        "waiting_time": "waiting_time",
        "waiting_hours": "waiting_hours",
        "waiting_amount": "waiting_amount",
        "add_pay": "add_pay",
        "paid_date": "paid_date",
        "job_outcome": "job_outcome",
        "description": "description",
        "hours": "hours",
    }

    insert_data = {}
    for api_field, db_field in field_map.items():
        if api_field in data:
            insert_data[db_field] = data[api_field]

    if not insert_data:
        raise HTTPException(status_code=400, detail="No valid fields supplied")

    columns = ", ".join(insert_data.keys())
    placeholders = ", ".join(["?"] * len(insert_data))
    values = list(insert_data.values())

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            f"INSERT INTO work_logs ({columns}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        row_id = cur.lastrowid

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()

    return {
        "message": "Job created successfully",
        "id": row_id,
    }


def update_job_record(job_id: str, job) -> dict:
    data = job.model_dump(exclude_unset=True)

    field_map = {
        "work_date": "work_date",
        "job_id": "job_id",
        "category": "category",
        "vehicle_description": "vehicle_description",
        "vehicle_reg": "vehicle_reg",
        "collection_from": "collection_from",
        "delivery_to": "delivery_to",
        "amount": "amount",
        "job_expenses": "job_expenses",
        "expenses_amount": "expenses_amount",
        "comments": "comments",
        "auth_code": "auth_code",
        "job_status": "job_status",
        "waiting_time": "waiting_time",
        "waiting_hours": "waiting_hours",
        "waiting_amount": "waiting_amount",
        "add_pay": "add_pay",
        "paid_date": "paid_date",
        "job_outcome": "job_outcome",
        "description": "description",
        "hours": "hours",
    }

    update_data = {}
    for api_field, db_field in field_map.items():
        if api_field in data:
            update_data[db_field] = data[api_field]

    if not update_data:
        raise HTTPException(status_code=400, detail="No valid fields supplied")

    set_clause = ", ".join([f"{col} = ?" for col in update_data.keys()])
    values = list(update_data.values())
    values.append(str(job_id))

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            f"UPDATE work_logs SET {set_clause} WHERE CAST(job_id AS TEXT) = ?",
            values,
        )
        conn.commit()

        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Job not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()

    return {"message": "Job updated successfully"}


def update_job_row_record(row_id: int, job) -> dict:
    data = job.model_dump(exclude_unset=True)

    field_map = {
        "work_date": "work_date",
        "job_id": "job_id",
        "category": "category",
        "vehicle_description": "vehicle_description",
        "vehicle_reg": "vehicle_reg",
        "collection_from": "collection_from",
        "delivery_to": "delivery_to",
        "amount": "amount",
        "job_expenses": "job_expenses",
        "expenses_amount": "expenses_amount",
        "comments": "comments",
        "auth_code": "auth_code",
        "job_status": "job_status",
        "waiting_time": "waiting_time",
        "waiting_hours": "waiting_hours",
        "waiting_amount": "waiting_amount",
        "add_pay": "add_pay",
        "paid_date": "paid_date",
        "job_outcome": "job_outcome",
        "description": "description",
        "hours": "hours",
    }

    update_data = {}
    for api_field, db_field in field_map.items():
        if api_field in data:
            update_data[db_field] = data[api_field]

    if not update_data:
        raise HTTPException(status_code=400, detail="No valid fields supplied")

    set_clause = ", ".join([f"{col} = ?" for col in update_data.keys()])
    values = list(update_data.values())
    values.append(row_id)

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            f"UPDATE work_logs SET {set_clause} WHERE id = ?",
            values,
        )
        conn.commit()

        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Job row not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()

    return {"message": "Job row updated successfully"}


def delete_job_record(job_id: str) -> dict:
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("DELETE FROM work_logs WHERE CAST(job_id AS TEXT) = ?", [str(job_id)])
        conn.commit()

        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Job not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()

    return {"message": "Job deleted successfully"}


def delete_job_row_record(row_id: int) -> dict:
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("DELETE FROM work_logs WHERE id = ?", [row_id])
        conn.commit()

        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Job row not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()

    return {"message": "Job row deleted successfully"}