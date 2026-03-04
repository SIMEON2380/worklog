# worklog/reporting.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import pandas as pd


@dataclass(frozen=True)
class Totals:
    total_job_amount: float = 0.0
    total_wait_hours: float = 0.0
    total_wait_amount: float = 0.0
    total_add_pay: float = 0.0
    driver_pay: float = 0.0
    total_expenses: float = 0.0
    total_received: float = 0.0


# -------------------------
# Week label helper
# -------------------------
def format_week_range(monday: date) -> str:
    sunday = monday + timedelta(days=6)
    return f"{monday:%d %b %Y} – {sunday:%d %b %Y}"


# -------------------------
# Month label helper
# -------------------------
def format_month_label(month_str: str) -> str:
    """Convert YYYY-MM → 'Mon YYYY'"""
    try:
        d = pd.to_datetime(f"{month_str}-01", errors="coerce")
        if pd.isna(d):
            return month_str
        return d.strftime("%b %Y")
    except Exception:
        return month_str


# -------------------------
# Totals calculation
# -------------------------
def compute_totals(df: pd.DataFrame) -> Totals:

    if df is None or df.empty:
        return Totals()

    def num(col: str) -> pd.Series:
        """Coerce numbers safely, handling '£', commas, and other text noise."""
        if col not in df.columns:
            return pd.Series([0.0] * len(df), index=df.index, dtype="float64")

        s = df[col]

        # Clean common money/text formats: "£60.00", "1,200", " 60 ", etc.
        if s.dtype == "object":
            s = (
                s.astype(str)
                .str.strip()
                .str.replace(",", "", regex=False)
                .str.replace("£", "", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)  # keep digits, dot, minus
            )

        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    job_amount = num("amount")
    wait_hours = num("waiting_hours")
    add_pay = num("add_pay")
    expenses = num("expenses_amount")

    # Waiting pay
    if "waiting_pay" in df.columns:
        wait_pay = num("waiting_pay")
    else:
        from .config import Config

        rate = float(getattr(Config(), "WAITING_RATE", 0.0))
        wait_pay = wait_hours * rate

    # Driver pay
    if "driver_pay" in df.columns:
        driver_pay = num("driver_pay")
    else:
        driver_pay = (job_amount + wait_pay + add_pay) - expenses

    # Total received
    if "total_received" in df.columns:
        received = num("total_received")
    else:
        received = (job_amount + add_pay + wait_pay) - expenses

    return Totals(
        total_job_amount=float(job_amount.sum()),
        total_wait_hours=float(wait_hours.sum()),
        total_wait_amount=float(wait_pay.sum()),
        total_add_pay=float(add_pay.sum()),
        driver_pay=float(driver_pay.sum()),
        total_expenses=float(expenses.sum()),
        total_received=float(received.sum()),
    )