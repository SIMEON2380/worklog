import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    APP_TITLE: str = "Worklog"

    # systemd: Environment=WORKLOG_DB_DIR=/var/lib/worklog
    DB_DIR: str = os.environ.get("WORKLOG_DB_DIR", "/var/lib/worklog")
    TABLE_NAME: str = "work_logs"

    WAITING_RATE: float = 7.50

    STATUS_OPTIONS = ["Start", "Completed", "Aborted", "Paid", "Pending", "Withdraw"]
    JOB_TYPE_OPTIONS = ["STRD Trade Plate", "Inspect and Collect", "Inspect and Collect 2"]

    # UI shows these, DB stores lower-case comma separated
    JOB_EXPENSE_OPTIONS = ["No expenses", "uber", "taxi", "train", "toll", "fuel", "other"]

    INSPECT_COLLECT_RATE: float = 8.00
    INSPECT_COLLECT_TYPES = {"Inspect and Collect", "Inspect and Collect 2"}

    # keep labels (including typos) to avoid breaking UI expectations
    UI_COLUMNS = [
        "Date",
        "job number",
        "job type",
        "vehcile description",
        "vehicle Reg",
        "collection from",
        "delivery to",
        "job amount",
        "Job Expenses",
        "expenses Amount",
        "Auth code",
        "job status",
        "waiting time",
        "comments",
    ]

    EXPECTED_DB_COLS = [
        "id",
        "work_date",
        "description",
        "hours",
        "amount",
        "job_id",
        "category",
        "job_status",
        "waiting_time",
        "waiting_hours",
        "waiting_amount",
        "vehicle_description",
        "vehicle_reg",
        "collection_from",
        "delivery_to",
        "job_expenses",
        "expenses_amount",
        "auth_code",
        "comments",
        "created_at",
    ]

    @property
    def DB_PATH(self) -> str:
        return os.path.join(self.DB_DIR, "worklog.db")