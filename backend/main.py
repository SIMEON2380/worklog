from fastapi import FastAPI, HTTPException, status

from backend.schemas import JobCreate, JobUpdate
from backend.services import (
    create_job_record,
    delete_job_record,
    get_job_by_id,
    list_jobs,
    update_job_record,
)

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
        return list_jobs()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return get_job_by_id(job_id)
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
        return create_job_record(job)
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
        return update_job_record(job_id, job)
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
        return delete_job_record(job_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )