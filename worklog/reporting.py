# worklog/reporting.py
from __future__ import annotations

from dataclasses import dataclass
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


def compute_totals(df: pd.DataFrame) -> Totals:
    """
    Centralised totals used by Daily/Weekly/Monthly reports.
    Returns a Totals dataclass so pages can use attribute access: t.total_job_amount.
    Safe if df is empty, missing columns, or numeric strings.
    """

    if df is None or df.empty:
        return Totals()

    def num(col: str) -> pd.Series:
        if col not in df.columns:
            # return a 0 series same length as df to keep math easy
            return pd.Series([0.0] * len(df), index=df.index, dtype="float64")
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Match your DB column names (adjust here if your DB uses different ones)
    job_amount = num("amount")
    wait_hours = num("waiting_hours")
    add_pay = num("add_pay")
    expenses = num("expenses_amount")
    received = num("total_received")  # if you store this; otherwise we compute it below

    # Waiting pay: if you store it, use it; else compute from cfg.WAITING_RATE elsewhere.
    wait_pay = num("waiting_pay")

    # If your DB does NOT store waiting_pay, compute it from hours * rate if a rate column exists.
    # (We won't import Config here to avoid circular imports.)
    if "waiting_pay" not in df.columns:
        # fallback: if you stored a per-hour rate in 'waiting_rate' use it, else assume 0
        rate = num("waiting_rate")
        wait_pay = wait_hours * rate

    # Driver pay: if stored use it; else (job_amount + wait_pay + add_pay - expenses) is a reasonable fallback
    driver_pay = num("driver_pay")
    if "driver_pay" not in df.columns:
        driver_pay = (job_amount + wait_pay + add_pay) - expenses

    # Total received: if not stored, compute it (job_amount + add_pay + wait_pay - expenses)
    if "total_received" not in df.columns:
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