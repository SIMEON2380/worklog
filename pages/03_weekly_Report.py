import streamlit as st
import pandas as pd
from datetime import date, timedelta

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, display_jobs_table
from worklog.reporting import compute_totals, format_week_range

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Weekly Report", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Weekly Report")


def actual_paid_received_in_week(df: pd.DataFrame, week_start: date) -> float:
    out = df.copy()

    if out.empty or "paid_date" not in out.columns:
        return 0.0

    status_col = "job_status" if "job_status" in out.columns else ("status" if "status" in out.columns else None)
    if not status_col:
        return 0.0

    out[status_col] = out[status_col].fillna("").astype(str).str.strip().str.lower()
    out["paid_date"] = pd.to_datetime(out["paid_date"], errors="coerce").dt.date

    week_end = week_start + timedelta(days=6)

    paid_df = out[
        (out[status_col] == "paid") &
        (out["paid_date"] >= week_start) &
        (out["paid_date"] <= week_end)
    ].copy()

    def _num(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0)

    total = (
        _num(paid_df.get("amount", pd.Series(dtype=float))).sum()
        + _num(paid_df.get("waiting_amount", pd.Series(dtype=float))).sum()
        + _num(paid_df.get("add_pay", pd.Series(dtype=float))).sum()
        - _num(paid_df.get("expenses_amount", pd.Series(dtype=float))).sum()
    )

    return round(float(total), 2)


df = DB["read_all"]()
today = date.today()
current_week_start = today - timedelta(days=today.weekday())

if df.empty:
    st.info("No jobs found.")
    st.write(f"Week starting: {current_week_start.isoformat()}")
    st.stop()

df = df.copy()

df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date
df = df.dropna(subset=["work_date"])

df["_week_start"] = df["work_date"].apply(lambda d: d - timedelta(days=d.weekday()))

all_weeks = sorted(df["_week_start"].unique().tolist(), reverse=True)
options = [current_week_start] + [w for w in all_weeks if w != current_week_start]

selected = st.selectbox(
    "Select week (Mon–Sun)",
    options,
    index=0,
    format_func=format_week_range,
)

sub = df[df["_week_start"] == selected].copy()
sub = sub.drop(columns=["_week_start"], errors="ignore")

st.caption(f"Showing jobs for: **{format_week_range(selected)}**")

for col in ["amount", "waiting_hours", "waiting_amount", "expenses_amount", "hours", "add_pay"]:
    if col in sub.columns:
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)

t = compute_totals(sub)
actual_paid = actual_paid_received_in_week(df, selected)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Job Amount", f"£{t.total_job_amount:,.2f}")
c2.metric("Waiting Pay", f"£{t.total_wait_amount:,.2f}")
c3.metric("Add-Pay", f"£{t.total_add_pay:,.2f}")
c4.metric("Driver Pay Estimate", f"£{t.driver_pay:,.2f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Waiting Hours", f"{t.total_wait_hours:,.2f} hrs")
c6.metric("Expenses (Reimbursed)", f"£{t.total_expenses:,.2f}")
c7.metric("Total Received", f"£{t.total_received:,.2f}")
c8.metric("Actually Paid This Week", f"£{actual_paid:,.2f}")

st.divider()

display_jobs_table(cfg, sub, caption="Jobs in selected week", show_paid_date=True)