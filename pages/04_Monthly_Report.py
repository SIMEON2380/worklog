import pandas as pd
from dataclasses import dataclass

@dataclass
class Totals:
    total_job_amount: float
    total_wait_hours: float
    total_wait_amount: float
    total_add_pay: float
    driver_pay: float
    total_expenses: float
    total_received: float


def compute_totals(df, waiting_rate=7.5):
    sub = df.copy()

    numeric_cols = [
        "amount",
        "waiting_hours",
        "waiting_amount",
        "expenses_amount",
        "hours",
        "add_pay",
    ]

    for col in numeric_cols:
        if col not in sub.columns:
            sub[col] = 0.0
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)

    total_job_amount = float(sub["amount"].sum())
    total_wait_hours = float(sub["waiting_hours"].sum())

    # Recalculate waiting_amount if missing or unreliable
    if "waiting_amount" in sub.columns:
        total_wait_amount = float(sub["waiting_amount"].sum())
    else:
        total_wait_amount = total_wait_hours * float(waiting_rate)

    total_add_pay = float(sub["add_pay"].sum()) if "add_pay" in sub.columns else 0.0
    total_expenses = float(sub["expenses_amount"].sum())

    driver_pay = total_job_amount + total_wait_amount + total_add_pay
    total_received = driver_pay - total_expenses

    return Totals(
        total_job_amount=total_job_amount,
        total_wait_hours=total_wait_hours,
        total_wait_amount=total_wait_amount,
        total_add_pay=total_add_pay,
        driver_pay=driver_pay,
        total_expenses=total_expenses,
        total_received=total_received,
    )