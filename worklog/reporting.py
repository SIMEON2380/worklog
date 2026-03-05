from __future__ import annotations
from typing import Dict, Any
import pandas as pd

NUM_COLS = ["amount", "waiting_hours", "waiting_amount", "expenses_amount"]

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUM_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out

def compute_totals(df: pd.DataFrame, waiting_rate: float, compute_waiting_if_missing: bool = True) -> Dict[str, Any]:
    """
    Returns totals dict:
      job_amount_total, waiting_hours_total, waiting_amount_total, expenses_total, grand_total
    """
    out = coerce_numeric(df)

    if compute_waiting_if_missing and "waiting_amount" in out.columns and "waiting_hours" in out.columns:
        calc_wait = out["waiting_hours"] * float(waiting_rate)
        # fill waiting_amount when it's 0 (common when older rows didn't store it)
        out["waiting_amount"] = out["waiting_amount"].where(out["waiting_amount"] != 0, calc_wait)

    job_amount_total = float(out["amount"].sum()) if "amount" in out.columns else 0.0
    waiting_hours_total = float(out["waiting_hours"].sum()) if "waiting_hours" in out.columns else 0.0
    waiting_amount_total = float(out["waiting_amount"].sum()) if "waiting_amount" in out.columns else 0.0
    expenses_total = float(out["expenses_amount"].sum()) if "expenses_amount" in out.columns else 0.0

    grand_total = job_amount_total + waiting_amount_total - expenses_total

    return {
        "job_amount_total": job_amount_total,
        "waiting_hours_total": waiting_hours_total,
        "waiting_amount_total": waiting_amount_total,
        "expenses_total": expenses_total,
        "grand_total": grand_total,
        "df": out,
    }