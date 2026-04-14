import os
from fastapi import FastAPI, Header, HTTPException, status

from backend.schemas import JobCreate, JobUpdate
from backend.services import (
    create_job_record,
    delete_job_record,
    delete_job_row_record,
    get_job_by_id,
    get_job_by_row_id,
    list_jobs,
    update_job_record,
    update_job_row_record,
)

app = FastAPI()

API_KEY = os.getenv("API_KEY")


def verify_api_key(x_api_key: str | None = Header(default=None)):
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured on the server"
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )


@app.get("/")
def root():
    return {"message": "Worklog API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/jobs")
def get_jobs(x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        return list_jobs()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        return get_job_by_id(job_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/jobs/row/{row_id}")
def get_job_row(row_id: int, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        return get_job_by_row_id(row_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/jobs", status_code=status.HTTP_201_CREATED)
def create_job(job: JobCreate, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        return create_job_record(job)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.put("/jobs/{job_id}")
def update_job(job_id: str, job: JobUpdate, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        return update_job_record(job_id, job)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.put("/jobs/row/{row_id}")
def update_job_row(row_id: int, job: JobUpdate, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        return update_job_row_record(row_id, job)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        return delete_job_record(job_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/jobs/row/{row_id}")
def delete_job_row(row_id: int, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        return delete_job_row_record(row_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
