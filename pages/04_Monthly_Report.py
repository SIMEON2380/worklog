import pandas as pd

def compute_totals(df):
    sub = df.copy()

    # Ensure numeric columns are numeric
    for col in ["amount", "waiting_hours", "waiting_amount", "expenses_amount", "hours"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)

    total_job_amount = float(sub["amount"].sum()) if "amount" in sub.columns else 0.0
    total_wait_hours = float(sub["waiting_hours"].sum()) if "waiting_hours" in sub.columns else 0.0
    total_wait_amount = float(sub["waiting_amount"].sum()) if "waiting_amount" in sub.columns else 0.0
    total_expenses = float(sub["expenses_amount"].sum()) if "expenses_amount" in sub.columns else 0.0

    # Example extra calculations (keep if you already use them)
    total_add_pay = 0.0
    driver_pay = total_job_amount + total_wait_amount
    total_received = driver_pay - total_expenses

    return type(
        "Totals",
        (),
        {
            "total_job_amount": total_job_amount,
            "total_wait_hours": total_wait_hours,
            "total_wait_amount": total_wait_amount,
            "total_add_pay": total_add_pay,
            "driver_pay": driver_pay,
            "total_expenses": total_expenses,
            "total_received": total_received,
        },
    )()