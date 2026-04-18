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
            detail="API key is not configured on the server",
        )

    if not x_api_key:
        logger.warning("Request missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    if x_api_key != API_KEY:
        logger.warning(f"Invalid API key attempt: {x_api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def resolve_service(*names: str):
    """
    Return the first matching function found in backend.services.
    This avoids crashes when function names differ slightly
    (e.g. create_job vs create_job_record).
    """
    for name in names:
        fn = getattr(services, name, None)
        if callable(fn):
            return fn

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"None of these service functions exist: {', '.join(names)}",
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

    list_jobs = resolve_service("list_jobs", "get_jobs", "read_jobs")

    if all_records:
        data = list_jobs(
            work_date=work_date,
            job_status=job_status,
            category=category,
            limit=1000000,
            page=1,
        )
        return {
            "data": data,
            "page": 1,
            "page_size": len(data),
            "count": len(data),
            "all_records": True,
        }

    data = list_jobs(
        work_date=work_date,
        job_status=job_status,
        category=category,
        limit=limit,
        page=page,
    )
    return {
        "data": data,
        "page": page,
        "page_size": limit,
        "count": len(data),
        "all_records": False,
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    get_job_fn = resolve_service("get_job", "get_job_by_id", "read_job")
    return get_job_fn(job_id)


@app.post("/jobs")
def create_job(payload: JobCreate, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    create_job_fn = resolve_service("create_job", "create_job_record", "insert_job")
    return create_job_fn(payload)


@app.put("/jobs/{job_id}")
def update_job(
    job_id: str,
    payload: JobUpdate,
    x_api_key: str | None = Header(default=None),
):
    verify_api_key(x_api_key)
    update_job_fn = resolve_service("update_job", "update_job_record", "edit_job")
    return update_job_fn(job_id, payload)


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    delete_job_fn = resolve_service("delete_job", "delete_job_record", "remove_job")
    return delete_job_fn(job_id)