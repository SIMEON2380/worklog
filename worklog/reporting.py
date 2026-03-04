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

    Key improvement:
    - Supports both DB-style column names (amount, add_pay, expenses_amount, waiting_hours)
      and UI/export-style names (job amount, Add-Pay, expenses Amount, waiting time, etc.)
    - Cleans currency/text values safely (£, commas, whitespace)
    - Can parse waiting time ranges like "10:30-11:30" into hours if waiting_hours not present
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
        """Return numeric series for the first matching column name among candidates."""
        for col in candidates:
            if col in df.columns:
                return _clean_numeric(df[col])
        return _zero()

    def waiting_hours_series() -> pd.Series:
        """
        Prefer a numeric waiting_hours column.
        If missing, try parsing 'waiting time' like '10:30-11:30' into hours.
        """
        if "waiting_hours" in df.columns:
            return _clean_numeric(df["waiting_hours"])

        # UI/export variants
        for col in ("waiting time", "Waiting time", "Waiting Time", "waiting_time"):
            if col in df.columns:
                s = df[col].astype(str).fillna("").str.strip()

                parts = s.str.split("-", n=1, expand=True)
                if parts.shape[1] < 2:
                    return _zero()

                start = pd.to_datetime(parts[0].str.strip(), format="%H:%M", errors="coerce")
                end = pd.to_datetime(parts[1].str.strip(), format="%H:%M", errors="coerce")

                hours = (end - start).dt.total_seconds() / 3600.0
                hours = hours.fillna(0.0)
                hours = hours.where(hours >= 0, 0.0)  # avoid negatives if format is bad
                return hours.astype("float64")

        return _zero()

    # ---- pull values with aliases (DB + UI) ----
    job_amount = num_any("amount", "job amount", "Job Amount")
    add_pay = num_any("add_pay", "Add-Pay", "add-pay", "add pay", "Add Pay", "additional_pay", "Additional Pay")
    expenses = num_any("expenses_amount", "expenses Amount", "Expenses Amount", "expenses amount")
    wait_hours = waiting_hours_series()

    # Waiting pay: use stored waiting_pay if present, else compute using Config.WAITING_RATE
    if any(c in df.columns for c in ("waiting_pay", "Waiting Pay", "waiting pay")):
        wait_pay = num_any("waiting_pay", "Waiting Pay", "waiting pay")
    else:
        from .config import Config

        rate = float(getattr(Config(), "WAITING_RATE", 0.0))
        wait_pay = wait_hours * rate

    # Driver pay: use stored driver_pay if present, else compute
    if any(c in df.columns for c in ("driver_pay", "Driver Pay", "driver pay")):
        driver_pay = num_any("driver_pay", "Driver Pay", "driver pay")
    else:
        driver_pay = (job_amount + wait_pay + add_pay) - expenses

    # Total received: use stored total_received if present, else compute
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