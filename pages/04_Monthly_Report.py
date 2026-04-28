import os
from datetime import date

import pandas as pd
import requests
import streamlit as st

from worklog.auth import ensure_default_user
from worklog.config import Config
from worklog.reporting import compute_totals, format_month_label
from worklog.ui import require_login, display_jobs_table

API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("WORKLOG_API_KEY", "")

cfg = Config()

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Monthly Report", layout="wide")

ensure_default_user(cfg)
require_login()

st.subheader("Monthly Report")


def fetch_jobs(params=None) -> list:
    response = requests.get(
        f"{API_URL}/jobs",
        headers={"x-api-key": API_KEY},
        params=params,
        timeout=15,
    )
    response.raise_for_status()

    payload = response.json()

    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]

    if isinstance(payload, list):
        return payload

    raise ValueError("Unexpected API response format")


today = date.today()
current_month = today.strftime("%Y-%m")

try:
    # First fetch: enough rows to build month selector
    records = fetch_jobs({"limit": 1000})
    df = pd.DataFrame(records)

    if df.empty:
        st.info("No jobs found.")
        st.write(f"Month: {current_month}")
        st.stop()

    if "work_date" not in df.columns:
        st.error("API response does not include 'work_date'.")
        st.write(df)
        st.stop()

    df = df.copy()
    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce")
    df = df.dropna(subset=["work_date"])

    if df.empty:
        st.info("No valid work dates found.")
        st.stop()

    df["_month"] = df["work_date"].dt.to_period("M").astype(str)

    all_months = sorted(df["_month"].unique().tolist(), reverse=True)

    if current_month in all_months:
        options = all_months
        default_index = all_months.index(current_month)
    else:
        options = [current_month] + all_months
        default_index = 0

    selected = st.selectbox(
        "Select month",
        options,
        index=default_index,
        format_func=format_month_label,
    )

    month_start = pd.Period(selected, freq="M").start_time.date()
    month_end = pd.Period(selected, freq="M").end_time.date()

    # Second fetch: fetch only selected month from API
    selected_records = fetch_jobs(
        {
            "start_date": month_start.isoformat(),
            "end_date": month_end.isoformat(),
            "limit": 2000,
        }
    )
    sub = pd.DataFrame(selected_records)

    st.caption(f"Showing jobs for: **{format_month_label(selected)}**")

    if sub.empty:
        st.info("No jobs found for this month.")
        st.stop()

    if "work_date" in sub.columns:
        sub["work_date"] = pd.to_datetime(sub["work_date"], errors="coerce").dt.date

    for col in ["amount", "waiting_hours", "waiting_amount", "expenses_amount", "hours", "add_pay"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)

    totals = compute_totals(sub)

except Exception as e:
    st.error(f"Failed to load jobs from API: {e}")
    st.stop()

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Job Amount", f"£{totals.total_job_amount:,.2f}")
c2.metric("Waiting Hours", f"{totals.total_wait_hours:,.2f} hrs")
c3.metric("Waiting Pay", f"£{totals.total_wait_amount:,.2f}")
c4.metric("Add-Pay", f"£{totals.total_add_pay:,.2f}")
c5.metric("Driver Pay", f"£{totals.driver_pay:,.2f}")
c6.metric("Expenses (Reimbursed)", f"£{totals.total_expenses:,.2f}")
c7.metric("Total Received", f"£{totals.total_received:,.2f}")

st.divider()

display_jobs_table(cfg, sub, caption=f"Jobs for {format_month_label(selected)}")