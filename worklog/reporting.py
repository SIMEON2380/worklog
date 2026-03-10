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


def _status_col(df: pd.DataFrame) -> Optional[str]:
    if "job_status" in df.columns:
        return "job_status"
    if "status" in df.columns:
        return "status"
    return None


def _normalise_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    status_col = _status_col(out)
    if status_col:
        out[status_col] = (
            out[status_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    return out


def _exclude_withdraw_for_pay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude only withdrawn jobs from pay calculations.
    All other statuses remain included.
    """
    out = _normalise_status(df)
    status_col = _status_col(out)
    if not status_col:
        return out
    return out[out[status_col] != "withdraw"].copy()


def compute_pending_totals(df: pd.DataFrame) -> float:
    """
    Returns total outstanding money for jobs with status = pending.

    Calculation:
        amount + waiting_amount + add_pay - expenses_amount
    """
    if df is None or getattr(df, "empty", True):
        return 0.0

    d = _normalise_status(df)
    status_col = _status_col(d)
    if not status_col:
        return 0.0

    pending_df = d[d[status_col] == "pending"].copy()
    if pending_df.empty:
        return 0.0

    pending_df["amount"] = (
        _num(pending_df["amount"])
        if "amount" in pending_df.columns
        else pd.Series([0.0] * len(pending_df), index=pending_df.index)
    )
    pending_df["waiting_hours"] = (
        _num(pending_df["waiting_hours"])
        if "waiting_hours" in pending_df.columns
        else pd.Series([0.0] * len(pending_df), index=pending_df.index)
    )
    pending_df["waiting_amount"] = (
        _num(pending_df["waiting_amount"])
        if "waiting_amount" in pending_df.columns
        else pd.Series([0.0] * len(pending_df), index=pending_df.index)
    )
    pending_df["expenses_amount"] = (
        _num(pending_df["expenses_amount"])
        if "expenses_amount" in pending_df.columns
        else pd.Series([0.0] * len(pending_df), index=pending_df.index)
    )
    pending_df["add_pay"] = (
        _num(pending_df["add_pay"])
        if "add_pay" in pending_df.columns
        else pd.Series([0.0] * len(pending_df), index=pending_df.index)
    )

    total = (
        float(pending_df["amount"].sum())
        + float(pending_df["waiting_amount"].sum())
        + float(pending_df["add_pay"].sum())
        - float(pending_df["expenses_amount"].sum())
    )

    return round(float(total), 2)


def compute_totals(
    df: pd.DataFrame,
    waiting_rate: float = 7.5,          # default keeps old pages working
    inspect_collect_rate: float = 8.0,  # safe default
    inspect_collect_types: Optional[Union[Iterable[str], set[str]]] = None,
) -> Totals:
    """
    Central totals used by Daily/Weekly/Monthly.

    Logic:
      - Withdraw jobs are excluded from pay totals
      - total_job_amount: sum(amount)
      - total_wait_hours: sum(waiting_hours)
      - total_wait_amount: sum(waiting_amount) if present else waiting_hours * waiting_rate
      - total_add_pay: sum(add_pay) + legacy Inspect & Collect auto-pay
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

    # Exclude withdrawn jobs from pay totals
    d = _exclude_withdraw_for_pay(df)

    if d.empty:
        return Totals(
            total_job_amount=0.0,
            total_wait_hours=0.0,
            total_wait_amount=0.0,
            total_add_pay=0.0,
            driver_pay=0.0,
            total_expenses=0.0,
            total_received=0.0,
        )

    # Coerce numeric columns safely
    d["amount"] = (
        _num(d["amount"])
        if "amount" in d.columns
        else pd.Series([0.0] * len(d), index=d.index)
    )
    d["waiting_hours"] = (
        _num(d["waiting_hours"])
        if "waiting_hours" in d.columns
        else pd.Series([0.0] * len(d), index=d.index)
    )
    d["waiting_amount"] = (
        _num(d["waiting_amount"])
        if "waiting_amount" in d.columns
        else pd.Series([0.0] * len(d), index=d.index)
    )
    d["expenses_amount"] = (
        _num(d["expenses_amount"])
        if "expenses_amount" in d.columns
        else pd.Series([0.0] * len(d), index=d.index)
    )
    d["hours"] = (
        _num(d["hours"])
        if "hours" in d.columns
        else pd.Series([0.0] * len(d), index=d.index)
    )
    d["add_pay"] = (
        _num(d["add_pay"])
        if "add_pay" in d.columns
        else pd.Series([0.0] * len(d), index=d.index)
    )

    # Totals
    total_job_amount = float(d["amount"].sum())
    total_wait_hours = float(d["waiting_hours"].sum())

    # Waiting pay: prefer stored waiting_amount; if missing/zero for rows, compute from hours
    wr = float(waiting_rate or 0.0)
    if "waiting_amount" in df.columns:
        computed_wait = d["waiting_hours"] * wr
        d["waiting_amount"] = d["waiting_amount"].where(d["waiting_amount"] != 0, computed_wait)
        total_wait_amount = float(d["waiting_amount"].sum())
    else:
        total_wait_amount = float((d["waiting_hours"] * wr).sum())

    # Manual add pay from column
    total_add_pay_manual = float(d["add_pay"].sum())

    # Legacy: Inspect & Collect auto add-pay
    default_types = {"Inspect and Collect", "Inspect and Collect 2"}
    types = set(inspect_collect_types) if inspect_collect_types else default_types

    total_add_pay_auto = 0.0
    if "category" in d.columns and types and inspect_collect_rate:
        mask = d["category"].fillna("").astype(str).isin([str(x) for x in types])
        total_add_pay_auto = float((d.loc[mask, "hours"] * float(inspect_collect_rate)).sum())

    # Combine manual + legacy auto add-pay
    total_add_pay = float(total_add_pay_manual + total_add_pay_auto)

    # Driver pay and received
    driver_pay = total_job_amount + total_wait_amount + total_add_pay
    total_expenses = float(d["expenses_amount"].sum())
    total_received = driver_pay + total_expenses

    return Totals(
        total_job_amount=round(total_job_amount, 2),
        total_wait_hours=round(total_wait_hours, 2),
        total_wait_amount=round(total_wait_amount, 2),
        total_add_pay=round(total_add_pay, 2),
        driver_pay=round(driver_pay, 2),
        total_expenses=round(total_expenses, 2),
        total_received=round(total_received, 2),
    )