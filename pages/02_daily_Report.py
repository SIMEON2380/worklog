import streamlit as st
import pandas as pd
from datetime import date

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, display_jobs_table
from worklog.reporting import compute_totals  # ✅ new

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Daily Report", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Daily Report")

df = DB["read_all"]()
today = date.today()

if df.empty:
    st.info("No jobs found.")
    st.write(f"Date: {today.isoformat()}")
    st.stop()

df = df.copy()
df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date
df = df.dropna(subset=["work_date"])

all_days = sorted(df["work_date"].unique().tolist(), reverse=True)
options = [today] + [d for d in all_days if d != today]

selected = st.selectbox("Select day", options, index=0)
sub = df[df["work_date"] == selected].copy()

# ✅ centralised totals (same for daily/weekly/monthly)
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

display_jobs_table(cfg, sub, caption="Jobs in selected day")