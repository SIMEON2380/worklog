import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query, status

from backend.schemas import JobCreate, JobUpdate
import backend.services as services

app = FastAPI()

API_KEY = os.getenv("API_KEY")


def verify_api_key(x_api_key: str | None = Header(default=None)):
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured on the server",
        )

    if x_api_key != API_KEY:
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
    job_outcome: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    limit: Optional[int] = Query(default=None, ge=1),
):
    verify_api_key(x_api_key)

    try:
        list_jobs_fn = resolve_service("list_jobs")

        kwargs = {}

        # Only pass arguments if they were provided, so we stay compatible
        # with different versions of services.py
        if work_date is not None:
            kwargs["work_date"] = work_date
        if job_status is not None:
            kwargs["job_status"] = job_status
        if job_outcome is not None:
            kwargs["job_outcome"] = job_outcome
        if category is not None:
            kwargs["category"] = category
        if search is not None:
            kwargs["search"] = search
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        if page is not None:
            kwargs["page"] = page
        if page_size is not None:
            kwargs["page_size"] = page_size
        if limit is not None:
            kwargs["limit"] = limit

        result = list_jobs_fn(**kwargs)

        # Normalize response shape
        if isinstance(result, dict):
            return result

        if isinstance(result, list):
            return {"data": result}

        return {"data": []}

    except TypeError:
        # Fallback for older/simpler services.py versions
        try:
            list_jobs_fn = resolve_service("list_jobs")

            if work_date is not None and limit is not None:
                result = list_jobs_fn(work_date=work_date, limit=limit)
            elif work_date is not None:
                result = list_jobs_fn(work_date=work_date)
            elif limit is not None:
                result = list_jobs_fn(limit=limit)
            else:
                result = list_jobs_fn()

            if isinstance(result, dict):
                return result
            if isinstance(result, list):
                return {"data": result}
            return {"data": []}

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    try:
        get_job_fn = resolve_service("get_job_by_id", "get_job")
        return get_job_fn(job_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/jobs", status_code=status.HTTP_201_CREATED)
def create_job(job: JobCreate, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    try:
        create_job_fn = resolve_service("create_job", "create_job_record")
        return create_job_fn(job)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.put("/jobs/{job_id}")
def update_job(job_id: str, job: JobUpdate, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    try:
        update_job_fn = resolve_service("update_job", "update_job_record")
        return update_job_fn(job_id, job)

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
        delete_job_fn = resolve_service("delete_job", "delete_job_record")
        return delete_job_fn(job_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/jobs/row/{row_id}")
def get_job_row(row_id: int, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    try:
        get_row_fn = resolve_service("get_job_by_row_id")
        return get_row_fn(row_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.put("/jobs/row/{row_id}")
def update_job_row(row_id: int, job: JobUpdate, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    try:
        update_row_fn = resolve_service("update_job_row_record", "update_job_row")
        return update_row_fn(row_id, job)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.delete("/jobs/row/{row_id}")
def delete_job_row(row_id: int, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    try:
        delete_row_fn = resolve_service("delete_job_row_record", "delete_job_row")
        return delete_row_fn(row_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )