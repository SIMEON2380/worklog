import streamlit as st
import pandas as pd
from datetime import date, timedelta

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, display_jobs_table
from worklog.reporting import compute_totals  # ✅ NEW

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Weekly Report", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Weekly Report")

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

selected = st.selectbox("Select week (Mon–Sun)", options, index=0)
sub = df[df["_week_start"] == selected].copy()
sub = sub.drop(columns=["_week_start"], errors="ignore")

# ✅ Centralised totals (same rules as Daily/Monthly)
t = compute_totals(sub)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Job Amount", f"£{t.total_job_amount:,.2f}")
c2.metric("Waiting Hours", f"{t.total_wait_hours:,.2f} hrs")
c3.metric("Waiting Pay", f"£{t.total_wait_amount:,.2f}")
c4.metric("Add-Pay", f"£{t.total_add_pay:,.2f}")
c5.metric("Driver Pay", f"£{t.driver_pay:,.2f}")
c6.metric("Expenses (Reimbursed)", f"£{t.total_expenses:,.2f}")
c7.metric("Total Received", f"£{t.total_received:,.2f}")

st.divider()

display_jobs_table(cfg, sub, caption="Jobs in selected week")