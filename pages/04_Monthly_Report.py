import streamlit as st
import pandas as pd
from datetime import date

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, display_jobs_table

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Monthly Report", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Monthly Report")

df = DB["read_all"]()
today = date.today()
current_month = today.strftime("%Y-%m")

if df.empty:
    st.info("No jobs found.")
    st.write(f"Month: {current_month}")
    st.stop()

df = df.copy()
dt = pd.to_datetime(df["work_date"], errors="coerce")
df["_month"] = dt.dt.to_period("M").astype(str)
df = df.dropna(subset=["_month"])

all_months = sorted(df["_month"].unique().tolist(), reverse=True)
options = [current_month] + [m for m in all_months if m != current_month]

selected = st.selectbox("Select month", options, index=0)
sub = df[df["_month"] == selected].copy()
sub = sub.drop(columns=["_month"], errors="ignore")

# -------------------------
# Totals
# -------------------------
total_job_amount = float(pd.to_numeric(sub["amount"], errors="coerce").fillna(0).sum())
total_wait_hours = float(pd.to_numeric(sub["waiting_hours"], errors="coerce").fillna(0).sum())
total_wait_amount = float(pd.to_numeric(sub["waiting_amount"], errors="coerce").fillna(0).sum())
total_expenses = float(pd.to_numeric(sub["expenses_amount"], errors="coerce").fillna(0).sum())

grand_total = total_job_amount + total_wait_amount - total_expenses

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Job Amount", f"£{total_job_amount:,.2f}")
c2.metric("Waiting Hours", f"{total_wait_hours:,.2f} hrs")
c3.metric("Waiting Pay", f"£{total_wait_amount:,.2f}")
c4.metric("Expenses", f"£{total_expenses:,.2f}")
c5.metric("Grand Total", f"£{grand_total:,.2f}")

st.divider()

display_jobs_table(cfg, sub, caption="Jobs in selected month")