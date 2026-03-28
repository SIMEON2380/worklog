from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from backend.db import get_connection

app = FastAPI()


class JobCreate(BaseModel):
    work_date: date
    job_id: str = Field(..., min_length=1, max_length=50)
    amount: float = Field(..., ge=0)
    job_status: Literal["Start", "Pending", "Paid", "Completed", "Aborted", "Withdraw"]


class JobUpdate(BaseModel):
    work_date: date
    amount: float = Field(..., ge=0)
    job_status: Literal["Start", "Pending", "Paid", "Completed", "Aborted", "Withdraw"]


@app.get("/")
def root():
    return {"message": "Worklog API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/jobs")
def get_jobs():
    try:
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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/jobs", status_code=status.HTTP_201_CREATED)
def create_job(job: JobCreate):
    try:
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.put("/jobs/{job_id}")
def update_job(job_id: str, job: JobUpdate):
    try:
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    try:
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )