from fastapi import HTTPException, status

from backend.db import get_connection


def list_jobs():
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
        ORDER BY id DESC
        LIMIT 20
    """)

    rows = cur.fetchall()
    conn.close()

    return [dict(row) for row in rows]


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
            detail="job not found"
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
            detail="job_id already exists"
        )

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
        job.job_id,
        job.amount,
        job.category,
        job.job_status,
        job.waiting_time,
        job.waiting_hours,
        job.waiting_amount,
        job.vehicle_description,
        job.vehicle_reg,
        job.collection_from,
        job.delivery_to,
        job.job_expenses,
        job.expenses_amount if job.expenses_amount is not None else 0.0,
        job.auth_code,
        job.comments,
        job.add_pay if job.add_pay is not None else 0.0,
        str(job.paid_date) if job.paid_date else None,
        job.job_outcome,
    ))

    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    return {
        "status": "success",
        "id": new_id,
        "job_id": job.job_id
    }


def update_job_record(job_id: str, job):
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
            detail="job not found"
        )

    cur.execute("""
        UPDATE work_logs
        SET
            work_date = ?,
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
        WHERE job_id = ?
    """, (
        str(job.work_date),
        job.amount,
        job.category,
        job.job_status,
        job.waiting_time,
        job.waiting_hours,
        job.waiting_amount,
        job.vehicle_description,
        job.vehicle_reg,
        job.collection_from,
        job.delivery_to,
        job.job_expenses,
        job.expenses_amount if job.expenses_amount is not None else 0.0,
        job.auth_code,
        job.comments,
        job.add_pay if job.add_pay is not None else 0.0,
        str(job.paid_date) if job.paid_date else None,
        job.job_outcome,
        job_id,
    ))

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "job_id": job_id
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
            detail="job not found"
        )

    cur.execute("""
        DELETE FROM work_logs
        WHERE job_id = ?
    """, (job_id,))

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "job_id": job_id
    }