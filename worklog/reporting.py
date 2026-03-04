# worklog/reporting.py
from __future__ import annotations

from typing import Dict
import pandas as pd


def compute_totals(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute numeric totals used by report pages.
    Safe on empty df / missing cols / string numbers.
    Returns a dict so pages can pick what they need.
    """

    if df is None or df.empty:
        return {
            "jobs_count": 0,
            "amount_total": 0.0,
            "expenses_total": 0.0,
            "net_total": 0.0,
        }

    def _sum_col(col: str) -> float:
        if col not in df.columns:
            return 0.0
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return float(s.sum())

    amount_total = _sum_col("amount")
    expenses_total = _sum_col("expenses_amount")

    return {
        "jobs_count": int(len(df)),
        "amount_total": amount_total,
        "expenses_total": expenses_total,
        "net_total": amount_total - expenses_total,
    }