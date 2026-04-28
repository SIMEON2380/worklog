import logging
import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query, status

from backend.schemas import JobCreate, JobUpdate
import backend.services as services

app = FastAPI()

logger = logging.getLogger("worklog.security")

API_KEY = os.getenv("WORKLOG_API_KEY") or os.getenv("API_KEY")


def verify_api_key(x_api_key: str | None = Header(default=None)):
    if not API_KEY:
        logger.error("API key not configured on server")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured on server",
        )

    if not x_api_key:
        logger.warning("Missing API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    if x_api_key != API_KEY:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized access attempt",
        )


@app.get("/")
def root():
    return {"message": "Worklog API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/jobs")
def get_jobs(
    x_api_key: str | None = Header(default=None),
    work_date: Optional[str] = Query(default=None),
    job_status: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=5000),
    page: int = Query(default=1, ge=1),
    all_records: bool = Query(default=False),
):
    verify_api_key(x_api_key)
    try:
        return services.list_jobs(
            work_date=work_date,
            job_status=job_status,
            category=category,
            limit=limit,
            page=page,
            all_records=all_records,
        )
    except HTTPException:
        raise
    except TypeError:
        return services.list_jobs()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        return services.get_job_by_id(job_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/jobs", status_code=status.HTTP_201_CREATED)
def create_job(payload: JobCreate, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        return services.create_job_record(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.put("/jobs/{job_id}")
def update_job(
    job_id: str,
    payload: JobUpdate,
    x_api_key: str | None = Header(default=None),
):
    verify_api_key(x_api_key)
    try:
        return services.update_job_record(job_id, payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        return services.delete_job_record(job_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )