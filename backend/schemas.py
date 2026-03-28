from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    work_date: date
    job_id: str = Field(..., min_length=1, max_length=50)
    amount: float = Field(..., ge=0)
    job_status: Literal["Start", "Pending", "Paid", "Completed", "Aborted", "Withdraw"]


class JobUpdate(BaseModel):
    work_date: date
    amount: float = Field(..., ge=0)
    job_status: Literal["Start", "Pending", "Paid", "Completed", "Aborted", "Withdraw"]