from datetime import date
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from backend.db import get_connection

app = FastAPI()


class JobCreate(BaseModel):
    work_date: date
    job_id: str = Field(..., min_length=1, max_length=50)
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
        return {"error": str(e)}


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
            return {"error": "job not found"}

        return dict(row)

    except Exception as e:
        return {"error": str(e)}


@app.post("/jobs")
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
            return {"error": "job_id already exists"}

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

    except Exception as e:
        return {"error": str(e)}