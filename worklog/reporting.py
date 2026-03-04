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
    """
    Centralised totals used by Daily/Weekly/Monthly reports.

    Key behaviour:
    - Handles numeric values stored as strings (e.g. "£60", "1,200")
    - Supports UI/export aliases for common columns
    - If add_pay column is missing, tries to extract Add-Pay from comment fields
      (e.g. "Add-pay 60", "add pay £60", "AddPay: 60")
    """

    if df is None or df.empty:
        return Totals()

    def _zero() -> pd.Series:
        return pd.Series([0.0] * len(df), index=df.index, dtype="float64")

    def _clean_numeric(series: pd.Series) -> pd.Series:
        s = series
        if s.dtype == "object":
            s = (
                s.astype(str)
                .str.strip()
                .str.replace(",", "", regex=False)
                .str.replace("£", "", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)  # keep digits, dot, minus
            )
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    def num_any(*candidates: str) -> pd.Series:
        """Return numeric series for first matching column name among candidates."""
        for col in candidates:
            if col in df.columns:
                return _clean_numeric(df[col])
        return _zero()

    def waiting_hours_series() -> pd.Series:
        """Prefer numeric waiting_hours; else parse waiting time like '10:30-11:30'."""
        if "waiting_hours" in df.columns:
            return _clean_numeric(df["waiting_hours"])

        for col in ("waiting time", "waiting_time", "Waiting time", "Waiting Time"):
            if col in df.columns:
                s = df[col].astype(str).fillna("").str.strip()
                parts = s.str.split("-", n=1, expand=True)
                if parts.shape[1] < 2:
                    return _zero()

                start = pd.to_datetime(parts[0].str.strip(), format="%H:%M", errors="coerce")
                end = pd.to_datetime(parts[1].str.strip(), format="%H:%M", errors="coerce")

                hours = (end - start).dt.total_seconds() / 3600.0
                hours = hours.fillna(0.0)
                hours = hours.where(hours >= 0, 0.0)
                return hours.astype("float64")

        return _zero()

    # --- Base numeric fields (DB + UI/export aliases) ---
    job_amount = num_any("amount", "job amount", "Job Amount")
    expenses = num_any("expenses_amount", "expenses Amount", "Expenses Amount", "expenses amount")

    # Waiting
    wait_hours = waiting_hours_series()

    # Waiting pay: stored or computed
    if any(c in df.columns for c in ("waiting_pay", "Waiting Pay", "waiting pay")):
        wait_pay = num_any("waiting_pay", "Waiting Pay", "waiting pay")
    else:
        from .config import Config

        rate = float(getattr(Config(), "WAITING_RATE", 0.0))
        wait_pay = wait_hours * rate

    # Add-Pay: stored column OR extracted from comments
    if any(c in df.columns for c in ("add_pay", "Add-Pay", "add-pay", "add pay", "Add Pay")):
        add_pay = num_any("add_pay", "Add-Pay", "add-pay", "add pay", "Add Pay")
    else:
        # Try extracting from comment-like fields
        comment_col = None
        for c in ("comment", "comments", "Comment", "Comments", "notes", "Notes", "description", "Description"):
            if c in df.columns:
                comment_col = c
                break

        if comment_col is None:
            add_pay = _zero()
        else:
            text = df[comment_col].astype(str).fillna("").str.lower()

            # Extract a number after patterns like:
            # "add pay 60", "add-pay £60", "addpay: 60.50"
            extracted = text.str.extract(
                r"add[\s\-]*pay[^0-9]*([0-9]+(?:\.[0-9]+)?)",
                expand=False,
            )

            add_pay = pd.to_numeric(extracted, errors="coerce").fillna(0.0)

    # Driver pay: stored or computed
    if any(c in df.columns for c in ("driver_pay", "Driver Pay", "driver pay")):
        driver_pay = num_any("driver_pay", "Driver Pay", "driver pay")
    else:
        driver_pay = (job_amount + wait_pay + add_pay) - expenses

    # Total received: stored or computed
    if any(c in df.columns for c in ("total_received", "Total Received", "total received")):
        received = num_any("total_received", "Total Received", "total received")
    else:
        received = (job_amount + wait_pay + add_pay) - expenses

    return Totals(
        total_job_amount=float(job_amount.sum()),
        total_wait_hours=float(wait_hours.sum()),
        total_wait_amount=float(wait_pay.sum()),
        total_add_pay=float(add_pay.sum()),
        driver_pay=float(driver_pay.sum()),
        total_expenses=float(expenses.sum()),
        total_received=float(received.sum()),
    )