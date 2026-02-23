import re
from typing import Any

STATUS_OPTIONS = ["Start", "Completed", "Aborted", "Paid", "Pending", "Withdraw"]
JOB_TYPE_OPTIONS = ["STRD Trade Plate", "Inspect and Collect", "Inspect and Collect 2"]
JOB_EXPENSE_OPTIONS = ["uber", "taxi", "train", "toll", "other"]

NON_PAYABLE_STATUSES = {"withdraw", "aborted"}


def normalize_status(x: Any) -> str:
    s = str(x or "").strip()
    return s if s in STATUS_OPTIONS else "Pending"


def normalize_job_type(x: Any) -> str:
    s = str(x or "").strip()
    return s if s in JOB_TYPE_OPTIONS else JOB_TYPE_OPTIONS[0]


def normalize_expense_type(x: Any) -> str:
    s = str(x or "").strip().lower()
    return s if s in JOB_EXPENSE_OPTIONS else "other"


def clean_job_number(val: Any) -> str:
    """
    Fix Excel numeric job ids like 11623733.0 -> 11623733
    Keeps text values as-is.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return ""
    s = s.replace(",", "")
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s.strip()


def zero_to_none(x):
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return None if abs(v) < 1e-12 else v