import os
from datetime import date

import pandas as pd
import requests
import streamlit as st

from worklog.config import Config
from worklog.auth import ensure_default_user
from worklog.ui import require_login
from worklog.reporting import compute_totals

API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("WORKLOG_API_KEY", "supersecret123")

cfg = Config()

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Monthly Report", layout="wide")

ensure_default_user(cfg)
require_login()

st.subheader("Monthly Report")

# -------------------------
# Load data from API
# -------------------------
try:
    response = requests.get(
        f"{API_URL}/jobs",
        headers={"x-api-key": API_KEY},
        timeout=15,
    )
    response.raise_for_status()

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

# -------------------------
# Prepare data
# -------------------------
today = date.today()
current_month = today.strftime("%Y-%m")

if df.empty:
    st.info("No jobs found.")
    st.write(f"Month: {current_month}")
    st.stop()

if "work_date" not in df.columns:
    st.error("Missing 'work_date' column in dataset.")
    st.stop()

df = df.copy()
df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce")
df = df.dropna(subset=["work_date"])

if df.empty:
    st.info("No valid work dates found.")
    st.stop()

df["_month"] = df["work_date"].dt.to_period("M").astype(str)
df["work_date"] = df["work_date"].dt.date

# -------------------------
# Month selector
# -------------------------
all_months = sorted(df["_month"].unique().tolist(), reverse=True)
options = [current_month] + [m for m in all_months if m != current_month]

selected = st.selectbox("Select month", options, index=0)

sub = df[df["_month"] == selected].copy()

st.write(f"Month: {selected}")

if sub.empty:
    st.info("No jobs found for this month.")
    st.stop()

# -------------------------
# Metrics
# -------------------------
totals = compute_totals(sub)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Job Amount", f"£{totals.total_job_amount:,.2f}")
c2.metric("Waiting Hours", f"{totals.total_wait_hours:,.2f} hrs")
c3.metric("Waiting Pay", f"£{totals.total_wait_amount:,.2f}")
c4.metric("Add-Pay", f"£{totals.total_add_pay:,.2f}")
c5.metric("Driver Pay", f"£{totals.driver_pay:,.2f}")
c6.metric("Expenses (Reimbursed)", f"£{totals.total_expenses:,.2f}")
c7.metric("Total Received", f"£{totals.total_received:,.2f}")

st.divider()

# -------------------------
# Table
# -------------------------
sub = sub.drop(columns=["_month"], errors="ignore")

st.dataframe(sub, use_container_width=True)
