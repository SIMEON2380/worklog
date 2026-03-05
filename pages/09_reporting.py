# worklog/reporting.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Union

import pandas as pd


# -------------------------
# Formatting helpers
# -------------------------
def format_week_range(week_start: date) -> str:
    """Input: a Monday date. Output: 'Mon 02 Mar 2026 – Sun 08 Mar 2026'."""
    if not isinstance(week_start, date):
        return str(week_start)
    week_end = week_start + timedelta(days=6)
    return f"{week_start.strftime('%a %d %b %Y')} – {week_end.strftime('%a %d %b %Y')}"


def format_month_label(month_key: str) -> str:
    """Input: 'YYYY-MM'. Output: 'March 2026'."""
    try:
        dt = datetime.strptime(month_key, "%Y-%m")
        return dt.strftime("%B %Y")
    except Exception:
        return str(month_key)


# -------------------------
# Totals model (what pages read)
# -------------------------
@dataclass(frozen=True)
class Totals:
    total_job_amount: float
    total_wait_hours: float
    total_wait_amount: float
    total_add_pay: float
    driver_pay: float
    total_expenses: float
    total_received: float


# -------------------------
# Core reporting
# -------------------------
def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def compute_totals(
    df: pd.DataFrame,
    waiting_rate: float = 7.5,          # default keeps old pages working
    inspect_collect_rate: float = 8.0,  # safe default
    inspect_collect_types: Optional[Union[Iterable[str], set[str]]] = None,
) -> Totals:
    """
    Central totals used by Daily/Weekly/Monthly.

    Logic:
      - total_job_amount: sum(amount)
      - total_wait_hours: sum(waiting_hours)
      - total_wait_amount: sum(waiting_amount) if present else waiting_hours * waiting_rate
      - total_add_pay: (NEW) sum(add_pay) + (legacy) Inspect & Collect auto-pay (hours * inspect_collect_rate)
      - driver_pay: job_amount + wait_amount + add_pay
      - total_received: driver_pay + expenses_amount (treated as reimbursed)
    """
    if df is None or getattr(df, "empty", True):
        return Totals(
            total_job_amount=0.0,
            total_wait_hours=0.0,
            total_wait_amount=0.0,
            total_add_pay=0.0,
            driver_pay=0.0,
            total_expenses=0.0,
            total_received=0.0,
        )

    d = df.copy()

    # Coerce numeric columns safely
    d["amount"] = _num(d["amount"]) if "amount" in d.columns else 0.0
    d["waiting_hours"] = _num(d["waiting_hours"]) if "waiting_hours" in d.columns else 0.0
    d["waiting_amount"] = _num(d["waiting_amount"]) if "waiting_amount" in d.columns else pd.Series([0.0] * len(d), index=d.index)
    d["expenses_amount"] = _num(d["expenses_amount"]) if "expenses_amount" in d.columns else 0.0
    d["hours"] = _num(d["hours"]) if "hours" in d.columns else 0.0

    # NEW: add_pay column (REAL DEFAULT 0)
    d["add_pay"] = _num(d["add_pay"]) if "add_pay" in d.columns else 0.0

    # Totals
    total_job_amount = float(pd.to_numeric(d["amount"], errors="coerce").fillna(0).sum())
    total_wait_hours = float(pd.to_numeric(d["waiting_hours"], errors="coerce").fillna(0).sum())

    # Waiting pay: prefer stored waiting_amount; if missing/zero for rows, compute from hours
    wr = float(waiting_rate or 0.0)
    if "waiting_amount" in df.columns:
        computed_wait = d["waiting_hours"] * wr
        d["waiting_amount"] = d["waiting_amount"].where(d["waiting_amount"] != 0, computed_wait)
        total_wait_amount = float(d["waiting_amount"].sum())
    else:
        total_wait_amount = float((d["waiting_hours"] * wr).sum())

    # NEW: manual add pay from column
    total_add_pay_manual = float(d["add_pay"].sum())

    # Legacy: Inspect & Collect auto add-pay (kept for backwards compatibility)
    default_types = {"Inspect and Collect", "Inspect and Collect 2"}
    types = set(inspect_collect_types) if inspect_collect_types else default_types

    total_add_pay_auto = 0.0
    if "category" in d.columns and types and inspect_collect_rate:
        mask = d["category"].astype(str).isin([str(x) for x in types])
        total_add_pay_auto = float((d.loc[mask, "hours"] * float(inspect_collect_rate)).sum())

    # Combine (so old Inspect&Collect still works + new Add Pay column works everywhere)
    total_add_pay = float(total_add_pay_manual + total_add_pay_auto)

    # Driver pay and received
    driver_pay = total_job_amount + total_wait_amount + total_add_pay
    total_expenses = float(d["expenses_amount"].sum())
    total_received = driver_pay + total_expenses

    return Totals(
        total_job_amount=total_job_amount,
        total_wait_hours=total_wait_hours,
        total_wait_amount=total_wait_amount,
        total_add_pay=total_add_pay,
        driver_pay=driver_pay,
        total_expenses=total_expenses,
        total_received=total_received,
    )