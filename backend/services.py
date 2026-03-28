from fastapi import HTTPException, status

from backend.db import get_connection


def list_jobs():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, work_date, job_id, amount, job_status
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
        SELECT id, work_date, job_id, amount, job_status
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
        INSERT INTO work_logs (work_date, job_id, amount, job_status)
        VALUES (?, ?, ?, ?)
    """, (str(job.work_date), job.job_id, job.amount, job.job_status))

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
        SET work_date = ?, amount = ?, job_status = ?
        WHERE job_id = ?
    """, (str(job.work_date), job.amount, job.job_status, job_id))

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