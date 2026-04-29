import logging
import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request, status

from backend.schemas import JobCreate, JobUpdate
import backend.services as services

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

logging.basicConfig(level=logging.INFO)

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

logger = logging.getLogger("worklog.security")

API_KEY = os.getenv("WORKLOG_API_KEY") or os.getenv("API_KEY")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path

    logger.info(f"{client_ip} -> {method} {path}")

    response = await call_next(request)

    logger.info(f"{client_ip} <- {response.status_code} {path}")

    return response


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
@limiter.limit("60/minute")
def get_jobs(
    request: Request,
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
@limiter.limit("60/minute")
def get_job(
    request: Request,
    job_id: str,
    x_api_key: str | None = Header(default=None),
):
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
@limiter.limit("30/minute")
def create_job(
    request: Request,
    payload: JobCreate,
    x_api_key: str | None = Header(default=None),
):
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


@app.put("/jobs/row/{row_id}")
@limiter.limit("30/minute")
def update_job_row(
    request: Request,
    row_id: int,
    payload: JobUpdate,
    x_api_key: str | None = Header(default=None),
):
    verify_api_key(x_api_key)
    try:
        return services.update_job_row_record(row_id, payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.delete("/jobs/row/{row_id}")
@limiter.limit("20/minute")
def delete_job_row(
    request: Request,
    row_id: int,
    x_api_key: str | None = Header(default=None),
):
    verify_api_key(x_api_key)
    try:
        return services.delete_job_row_record(row_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.put("/jobs/{job_id}")
@limiter.limit("30/minute")
def update_job(
    request: Request,
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
@limiter.limit("20/minute")
def delete_job(
    request: Request,
    job_id: str,
    x_api_key: str | None = Header(default=None),
):
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