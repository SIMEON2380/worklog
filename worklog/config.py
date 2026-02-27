from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    APP_TITLE: str = "Worklog"

    # systemd: Environment=WORKLOG_DB_DIR=/var/lib/worklog
    DB_DIR: str = os.environ.get("WORKLOG_DB_DIR", "/var/lib/worklog")
    DB_FILE: str = "worklog.sqlite3"

    TABLE_NAME: str = "work_logs"
    USERS_TABLE: str = "users"

    WAITING_RATE: float = 7.50

    # Your existing options (kept)
    STATUS_OPTIONS = ["Start", "Completed", "Aborted", "Paid", "Pending", "Withdraw"]
    JOB_TYPE_OPTIONS = ["STRD Trade Plate", "Inspect and Collect", "Inspect and Collect 2"]
    JOB_EXPENSE_OPTIONS = ["uber", "taxi", "train", "toll", "other"]

    # Inspect & Collect pay rate
    INSPECT_COLLECT_RATE: float = 8.00
    INSPECT_COLLECT_TYPES = {"Inspect and Collect", "Inspect and Collect 2"}

    # Security
    PASSWORD_MIN_LEN: int = 6