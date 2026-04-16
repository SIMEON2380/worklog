from fastapi import HTTPException, status

from backend.db import get_connection

WAITING_RATE = 7.50


def normalise_text(value):
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def normalise_vehicle_description(value):
    value = normalise_text(value)
    return value.upper() if value else None


def normalise_job_expenses(value):
    value = normalise_text(value)
    if not value:
        return "no expenses"
    return value.lower()


def parse_wait_range_to_hours(s: str) -> float:
    if not s:
        return 0.0

    s = str(s).strip().replace(" ", "")
    if "-" not in s:
        return 0.0

    start, end = s.split("-", 1)

    def to_minutes(value: str) -> int:
        if ":" in value:
            h, m = value.split(":")
            return int(h) * 60 + int(m)
        return int(value) * 60

    try:
        start_m = to_minutes(start)
        end_m = to_minutes(end)
    except Exception:
        return 0.0

    if end_m <= start_m:
        return 0.0

    return round((end_m - start_m) / 60, 2)


def list_jobs(
    job_status=None,
    job_outcome=None,
    category=None,
    search=None,
    start_date=None,
    end_date=None,
    page=1,
    page_size=50,
):
    conn = get_connection()
    cur = conn.cursor()

    base_query = """
        FROM work_logs
        WHERE 1=1
    """
    params = []

    if job_status:
        base_query += " AND LOWER(TRIM(COALESCE(job_status, ''))) = LOWER(TRIM(?))"
        params.append(job_status)

    if job_outcome:
        base_query += " AND LOWER(TRIM(COALESCE(job_outcome, ''))) = LOWER(TRIM(?))"
        params.append(job_outcome)

    if category:
        base_query += " AND LOWER(TRIM(COALESCE(category, ''))) = LOWER(TRIM(?))"
        params.append(category)

    if start_date:
        base_query += " AND work_date >= ?"
        params.append(start_date)

    if end_date:
        base_query += " AND work_date <= ?"
        params.append(end_date)

    if search:
        search_value = f"%{str(search).strip()}%"
        base_query += """
            AND (
                COALESCE(job_id, '') LIKE ?
                OR COALESCE(vehicle_reg, '') LIKE ?
                OR COALESCE(vehicle_description, '') LIKE ?
                OR COALESCE(collection_from, '') LIKE ?
                OR COALESCE(delivery_to, '') LIKE ?
            )
        """
        params.extend([
            search_value,
            search_value,
            search_value,
            search_value,
            search_value,
        ])

    count_query = f"SELECT COUNT(*) {base_query}"
    cur.execute(count_query, params)
    total = cur.fetchone()[0]

    offset = (page - 1) * page_size

    data_query = f"""
        SELECT
            id,
            work_date,
            job_id,
            amount,
            category,
            job_status,
            waiting_time,
            waiting_hours,
            waiting_amount,
            vehicle_description,
            vehicle_reg,
            collection_from,
            delivery_to,
            job_expenses,
            expenses_amount,
            auth_code,
            comments,
            add_pay,
            paid_date,
            job_outcome
        {base_query}
        ORDER BY work_date DESC, id DESC
        LIMIT ? OFFSET ?
    """

    data_params = params + [page_size, offset]
    cur.execute(data_query, data_params)
    rows = cur.fetchall()
    conn.close()

    data = [dict(row) for row in rows]
    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 1

    return {
        "data": data,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
    }


def get_job_by_id(job_id: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id,
            work_date,
            job_id,
            amount,
            category,
            job_status,
            waiting_time,
            waiting_hours,
            waiting_amount,
            vehicle_description,
            vehicle_reg,
            collection_from,
            delivery_to,
            job_expenses,
            expenses_amount,
            auth_code,
            comments,
            add_pay,
            paid_date,
            job_outcome
        FROM work_logs
        WHERE job_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (job_id,))

    row = cur.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="job not found",
        )

    return dict(row)


def create_job_record(job):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 1
        FROM work_logs
        WHERE job_id = ?
        LIMIT 1
    """, (job.job_id,))

    existing = cur.fetchone()
    if existing:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="job_id already exists",
        )

    waiting_time = normalise_text(job.waiting_time)
    waiting_hours = parse_wait_range_to_hours(waiting_time or "")
    waiting_amount = round(waiting_hours * WAITING_RATE, 2)

    cur.execute("""
        INSERT INTO work_logs (
            work_date,
            job_id,
            amount,
            category,
            job_status,
            waiting_time,
            waiting_hours,
            waiting_amount,
            vehicle_description,
            vehicle_reg,
            collection_from,
            delivery_to,
            job_expenses,
            expenses_amount,
            auth_code,
            comments,
            add_pay,
            paid_date,
            job_outcome
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(job.work_date),
        normalise_text(job.job_id),
        job.amount if job.amount is not None else 0.0,
        normalise_text(job.category),
        normalise_text(job.job_status) or "Start",
        waiting_time,
        waiting_hours,
        waiting_amount,
        normalise_vehicle_description(job.vehicle_description),
        normalise_text(job.vehicle_reg),
        normalise_text(job.collection_from),
        normalise_text(job.delivery_to),
        normalise_job_expenses(job.job_expenses),
        job.expenses_amount if job.expenses_amount is not None else 0.0,
        normalise_text(job.auth_code),
        normalise_text(job.comments),
        job.add_pay if job.add_pay is not None else 0.0,
        str(job.paid_date) if job.paid_date else None,
        normalise_text(job.job_outcome),
    ))

    conn.commit()
    new_id = cur.lastrowid

    cur.execute("""
        SELECT
            id,
            work_date,
            job_id,
            amount,
            category,
            job_status,
            waiting_time,
            waiting_hours,
            waiting_amount,
            vehicle_description,
            vehicle_reg,
            collection_from,
            delivery_to,
            job_expenses,
            expenses_amount,
            auth_code,
            comments,
            add_pay,
            paid_date,
            job_outcome
        FROM work_logs
        WHERE id = ?
    """, (new_id,))

    row = cur.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inserted row could not be retrieved",
        )

    return {
        "status": "success",
        "data": dict(row),
    }


def update_job_record(job_id: str, job):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id,
            work_date,
            job_id,
            amount,
            category,
            job_status,
            waiting_time,
            waiting_hours,
            waiting_amount,
            vehicle_description,
            vehicle_reg,
            collection_from,
            delivery_to,
            job_expenses,
            expenses_amount,
            auth_code,
            comments,
            add_pay,
            paid_date,
            job_outcome
        FROM work_logs
        WHERE job_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (job_id,))

    existing = cur.fetchone()
    if not existing:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="job not found",
        )

    existing = dict(existing)
    updates = job.model_dump(exclude_unset=True)
    merged = {**existing, **updates}

    waiting_time = normalise_text(merged.get("waiting_time"))
    waiting_hours = parse_wait_range_to_hours(waiting_time or "")
    waiting_amount = round(waiting_hours * WAITING_RATE, 2)

    cur.execute("""
        UPDATE work_logs
        SET
            work_date = ?,
            job_id = ?,
            amount = ?,
            category = ?,
            job_status = ?,
            waiting_time = ?,
            waiting_hours = ?,
            waiting_amount = ?,
            vehicle_description = ?,
            vehicle_reg = ?,
            collection_from = ?,
            delivery_to = ?,
            job_expenses = ?,
            expenses_amount = ?,
            auth_code = ?,
            comments = ?,
            add_pay = ?,
            paid_date = ?,
            job_outcome = ?
        WHERE id = ?
    """, (
        str(merged.get("work_date")),
        normalise_text(merged.get("job_id")),
        merged.get("amount") if merged.get("amount") is not None else 0.0,
        normalise_text(merged.get("category")),
        normalise_text(merged.get("job_status")) or "Start",
        waiting_time,
        waiting_hours,
        waiting_amount,
        normalise_vehicle_description(merged.get("vehicle_description")),
        normalise_text(merged.get("vehicle_reg")),
        normalise_text(merged.get("collection_from")),
        normalise_text(merged.get("delivery_to")),
        normalise_job_expenses(merged.get("job_expenses")),
        merged.get("expenses_amount") if merged.get("expenses_amount") is not None else 0.0,
        normalise_text(merged.get("auth_code")),
        normalise_text(merged.get("comments")),
        merged.get("add_pay") if merged.get("add_pay") is not None else 0.0,
        str(merged.get("paid_date")) if merged.get("paid_date") else None,
        normalise_text(merged.get("job_outcome")),
        existing["id"],
    ))

    conn.commit()

    cur.execute("""
        SELECT
            id,
            work_date,
            job_id,
            amount,
            category,
            job_status,
            waiting_time,
            waiting_hours,
            waiting_amount,
            vehicle_description,
            vehicle_reg,
            collection_from,
            delivery_to,
            job_expenses,
            expenses_amount,
            auth_code,
            comments,
            add_pay,
            paid_date,
            job_outcome
        FROM work_logs
        WHERE id = ?
    """, (existing["id"],))

    row = cur.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="job not found after update",
        )

    return {
        "status": "success",
        "data": dict(row),
    }


def delete_job_record(job_id: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 1
        FROM work_logs
        WHERE job_id = ?
        LIMIT 1
    """, (job_id,))

    existing = cur.fetchone()
    if not existing:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="job not found",
        )

    cur.execute("""
        DELETE FROM work_logs
        WHERE job_id = ?
    """, (job_id,))

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "job_id": job_id,
    }


def get_job_by_row_id(row_id: int):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id,
            work_date,
            job_id,
            amount,
            category,
            job_status,
            waiting_time,
            waiting_hours,
            waiting_amount,
            vehicle_description,
            vehicle_reg,
            collection_from,
            delivery_to,
            job_expenses,
            expenses_amount,
            auth_code,
            comments,
            add_pay,
            paid_date,
            job_outcome
        FROM work_logs
        WHERE id = ?
        LIMIT 1
    """, (row_id,))

    row = cur.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="job not found",
        )

    return dict(row)


def update_job_row_record(row_id: int, job):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id,
            work_date,
            job_id,
            amount,
            category,
            job_status,
            waiting_time,
            waiting_hours,
            waiting_amount,
            vehicle_description,
            vehicle_reg,
            collection_from,
            delivery_to,
            job_expenses,
            expenses_amount,
            auth_code,
            comments,
            add_pay,
            paid_date,
            job_outcome
        FROM work_logs
        WHERE id = ?
        LIMIT 1
    """, (row_id,))

    existing = cur.fetchone()
    if not existing:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="row not found",
        )

    existing = dict(existing)
    updates = job.model_dump(exclude_unset=True)
    merged = {**existing, **updates}

    waiting_time = normalise_text(merged.get("waiting_time"))
    waiting_hours = parse_wait_range_to_hours(waiting_time or "")
    waiting_amount = round(waiting_hours * WAITING_RATE, 2)

    cur.execute("""
        UPDATE work_logs
        SET
            work_date = ?,
            job_id = ?,
            amount = ?,
            category = ?,
            job_status = ?,
            waiting_time = ?,
            waiting_hours = ?,
            waiting_amount = ?,
            vehicle_description = ?,
            vehicle_reg = ?,
            collection_from = ?,
            delivery_to = ?,
            job_expenses = ?,
            expenses_amount = ?,
            auth_code = ?,
            comments = ?,
            add_pay = ?,
            paid_date = ?,
            job_outcome = ?
        WHERE id = ?
    """, (
        str(merged.get("work_date")),
        normalise_text(merged.get("job_id")),
        merged.get("amount") if merged.get("amount") is not None else 0.0,
        normalise_text(merged.get("category")),
        normalise_text(merged.get("job_status")) or "Start",
        waiting_time,
        waiting_hours,
        waiting_amount,
        normalise_vehicle_description(merged.get("vehicle_description")),
        normalise_text(merged.get("vehicle_reg")),
        normalise_text(merged.get("collection_from")),
        normalise_text(merged.get("delivery_to")),
        normalise_job_expenses(merged.get("job_expenses")),
        merged.get("expenses_amount") if merged.get("expenses_amount") is not None else 0.0,
        normalise_text(merged.get("auth_code")),
        normalise_text(merged.get("comments")),
        merged.get("add_pay") if merged.get("add_pay") is not None else 0.0,
        str(merged.get("paid_date")) if merged.get("paid_date") else None,
        normalise_text(merged.get("job_outcome")),
        row_id,
    ))

    conn.commit()

    cur.execute("""
        SELECT
            id,
            work_date,
            job_id,
            amount,
            category,
            job_status,
            waiting_time,
            waiting_hours,
            waiting_amount,
            vehicle_description,
            vehicle_reg,
            collection_from,
            delivery_to,
            job_expenses,
            expenses_amount,
            auth_code,
            comments,
            add_pay,
            paid_date,
            job_outcome
        FROM work_logs
        WHERE id = ?
        LIMIT 1
    """, (row_id,))

    row = cur.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="row not found after update",
        )

    return {
        "status": "success",
        "data": dict(row),
    }


def delete_job_row_record(row_id: int):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 1
        FROM work_logs
        WHERE id = ?
        LIMIT 1
    """, (row_id,))

    existing = cur.fetchone()
    if not existing:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="row not found",
        )

    cur.execute("""
        DELETE FROM work_logs
        WHERE id = ?
    """, (row_id,))

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "row_id": row_id,
    }