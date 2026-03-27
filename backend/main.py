from fastapi import FastAPI
from backend.db import get_connection

app = FastAPI()


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