import streamlit as st
import pandas as pd
import requests
from datetime import date

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, display_jobs_table
from worklog.reporting import compute_totals

API_URL = "http://127.0.0.1:8000"
API_KEY = "supersecret123"

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Daily Report", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Daily Report")

try:
    response = requests.get(
        f"{API_URL}/jobs",
        headers={"x-api-key": API_KEY},
        timeout=15,
    )

    if response.status_code != 200:
        st.error(f"API failed: {response.status_code}")
        st.write(response.text)
        st.stop()

    payload = response.json()

    if isinstance(payload, dict) and "data" in payload:
        records = payload["data"]
    elif isinstance(payload, list):
        records = payload
    else:
        st.error("Unexpected API response format.")
        st.write(payload)
        st.stop()

    df = pd.DataFrame(records)

except Exception as e:
    st.error(f"Failed to load jobs from API: {e}")
    st.stop()

today = date.today()

if df.empty:
    st.info("No jobs found.")
    st.write(f"Date: {today.isoformat()}")
    st.stop()

if "work_date" not in df.columns:
    st.error("API response does not include 'work_date'.")
    st.write(df)
    st.stop()

df = df.copy()
df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date
df = df.dropna(subset=["work_date"])

all_days = sorted(df["work_date"].unique().tolist(), reverse=True)
options = [today] + [d for d in all_days if d != today]

selected = st.selectbox("Select day", options, index=0)
sub = df[df["work_date"] == selected].copy()

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