from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    work_date: date
    job_id: str = Field(..., min_length=1, max_length=50)
    amount: float = Field(..., ge=0)
    job_status: Literal["Start", "Pending", "Paid", "Completed", "Aborted", "Withdraw"]

    category: Optional[str] = None
    waiting_time: Optional[str] = None
    waiting_hours: Optional[float] = None
    waiting_amount: Optional[float] = None
    vehicle_description: Optional[str] = None
    vehicle_reg: Optional[str] = None
    collection_from: Optional[str] = None
    delivery_to: Optional[str] = None
    job_expenses: Optional[str] = None
    expenses_amount: Optional[float] = 0.0
    auth_code: Optional[str] = None
    comments: Optional[str] = None
    add_pay: Optional[float] = 0.0
    paid_date: Optional[date] = None
    job_outcome: Optional[str] = None


class JobUpdate(BaseModel):
    work_date: date
    amount: float = Field(..., ge=0)
    job_status: Literal["Start", "Pending", "Paid", "Completed", "Aborted", "Withdraw"]

    category: Optional[str] = None
    waiting_time: Optional[str] = None
    waiting_hours: Optional[float] = None
    waiting_amount: Optional[float] = None
    vehicle_description: Optional[str] = None
    vehicle_reg: Optional[str] = None
    collection_from: Optional[str] = None
    delivery_to: Optional[str] = None
    job_expenses: Optional[str] = None
    expenses_amount: Optional[float] = 0.0
    auth_code: Optional[str] = None
    comments: Optional[str] = None
    add_pay: Optional[float] = 0.0
    paid_date: Optional[date] = None
    job_outcome: Optional[str] = None