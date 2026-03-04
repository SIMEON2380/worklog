# worklog/reporting.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

# Columns we will scan for add-pay text (covers your common variants)
COMMENT_COL_CANDIDATES: List[str] = [
    "comment", "comments", "Comment", "Comments",
    "note", "notes", "Note", "Notes",
]

# Add-Pay parser: expects clean format "add-pay 15" / "add pay: £12.50" / "addpay=5"
ADD_PAY_RE = re.compile(
    r"(?:add[\s_-]*pay|addpay)\s*[:=]?\s*£?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

def extract_add_pay(text: Any) -> float:
    """Extract add-pay numbers from a free-text cell. Always positive."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0.0
    s = str(text)
    matches = ADD_PAY_RE.findall(s)
    if not matches:
        return 0.0
    total = 0.0
    for m in matches:
        try:
            total += abs(float(m))
        except ValueError:
            pass
    return float(total)

def add_pay_total(df: pd.DataFrame) -> float:
    """Sum add-pay across ALL available comment/note columns."""
    if df is None or df.empty:
        return 0.0

    cols = [c for c in COMMENT_COL_CANDIDATES if c in df.columns]
    if not cols:
        return 0.0

    per_row = pd.Series(0.0, index=df.index, dtype="float64")
    for c in cols:
        per_row = per_row.add(df[c].apply(extract_add_pay), fill_value=0.0)

    return float(pd.to_numeric(per_row, errors="coerce").fillna(0).sum())

def nsum(df: pd.DataFrame, col: str) -> float:
    """Numeric sum with coercion; returns 0.0 if column missing."""
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

@dataclass(frozen=True)
class Totals:
    total_job_amount: float
    total_wait_hours: float
    total_wait_amount: float
    total_expenses: float
    total_add_pay: float
    driver_pay: float
    total_received: float

def compute_totals(sub: pd.DataFrame) -> Totals:
    """
    Central totals rule-set.
    - Expenses are reimbursed (NOT subtracted from earnings)
    - Add-pay parsed from comment fields
    """
    total_job_amount = nsum(sub, "amount")
    total_wait_hours = nsum(sub, "waiting_hours")
    total_wait_amount = nsum(sub, "waiting_amount")
    total_expenses = nsum(sub, "expenses_amount")
    total_add_pay = add_pay_total(sub)

    driver_pay = total_job_amount + total_wait_amount + total_add_pay
    total_received = driver_pay + total_expenses

    return Totals(
        total_job_amount=total_job_amount,
        total_wait_hours=total_wait_hours,
        total_wait_amount=total_wait_amount,
        total_expenses=total_expenses,
        total_add_pay=total_add_pay,
        driver_pay=driver_pay,
        total_received=total_received,
    )