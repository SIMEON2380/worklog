from typing import Optional

from pydantic import BaseModel, field_validator


STATUS_OPTIONS = ["Start", "Completed", "Aborted", "Paid", "Pending", "Withdraw"]
JOB_TYPES = [
    "STRD Trade Plate",
    "Inspect and Collect",
    "Inspect and Collect 2",
]


def validate_status_value(value: Optional[str]) -> str:
    if value is None:
        return "Start"
    value = str(value).strip()
    if value not in STATUS_OPTIONS:
        raise ValueError(f"Invalid job_status. Must be one of {STATUS_OPTIONS}")
    return value


def validate_category_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if value not in JOB_TYPES:
        raise ValueError(f"Invalid category. Must be one of {JOB_TYPES}")
    return value


def validate_non_negative(value, field_name: str) -> float:
    if value is None:
        return 0.0
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return float(value)


def validate_waiting_time_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if "-" not in value:
        raise ValueError("waiting_time must be in format 'HH:MM-HH:MM' or 'HH-HH'")
    return value


class JobCreate(BaseModel):
    work_date: str
    job_id: str
    amount: float = 0.0
    category: Optional[str] = None
    job_status: Optional[str] = "Start"
    waiting_time: Optional[str] = None
    waiting_hours: Optional[float] = 0.0
    waiting_amount: Optional[float] = 0.0
    vehicle_description: Optional[str] = None
    vehicle_reg: Optional[str] = None
    collection_from: Optional[str] = None
    delivery_to: Optional[str] = None
    job_expenses: Optional[str] = None
    expenses_amount: Optional[float] = 0.0
    auth_code: Optional[str] = None
    comments: Optional[str] = None
    add_pay: Optional[float] = 0.0
    paid_date: Optional[str] = None
    job_outcome: Optional[str] = None

    @field_validator("job_status")
    @classmethod
    def validate_job_status(cls, value):
        return validate_status_value(value)

    @field_validator("category")
    @classmethod
    def validate_category(cls, value):
        return validate_category_value(value)

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value):
        return validate_non_negative(value, "amount")

    @field_validator("expenses_amount")
    @classmethod
    def validate_expenses_amount(cls, value):
        return validate_non_negative(value, "expenses_amount")

    @field_validator("add_pay")
    @classmethod
    def validate_add_pay(cls, value):
        return validate_non_negative(value, "add_pay")

    @field_validator("waiting_time")
    @classmethod
    def validate_waiting_time(cls, value):
        return validate_waiting_time_value(value)
class JobUpdate(BaseModel):
    work_date: Optional[str] = None
    job_id: Optional[str] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    job_status: Optional[str] = None
    waiting_time: Optional[str] = None
    waiting_hours: Optional[float] = None
    waiting_amount: Optional[float] = None
    vehicle_description: Optional[str] = None
    vehicle_reg: Optional[str] = None
    collection_from: Optional[str] = None
    delivery_to: Optional[str] = None
    job_expenses: Optional[str] = None
    expenses_amount: Optional[float] = None
    auth_code: Optional[str] = None
    comments: Optional[str] = None
    add_pay: Optional[float] = None
    paid_date: Optional[str] = None
    job_outcome: Optional[str] = None

    @field_validator("job_status")
    @classmethod
    def validate_job_status(cls, value):
        return validate_status_value(value)

    @field_validator("category")
    @classmethod
    def validate_category(cls, value):
        return validate_category_value(value)

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value):
        return validate_non_negative(value, "amount")

    @field_validator("expenses_amount")
    @classmethod
    def validate_expenses_amount(cls, value):
        return validate_non_negative(value, "expenses_amount")

    @field_validator("add_pay")
    @classmethod
    def validate_add_pay(cls, value):
        return validate_non_negative(value, "add_pay")

    @field_validator("waiting_time")
    @classmethod
    def validate_waiting_time(cls, value):
        return validate_waiting_time_value(value)
