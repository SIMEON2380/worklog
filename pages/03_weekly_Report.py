import os
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

from worklog.auth import ensure_default_user
from worklog.config import Config
from worklog.reporting import compute_totals, format_week_range
from worklog.ui import require_login, display_jobs_table

API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("WORKLOG_API_KEY", "supersecret123")

cfg = Config()

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Weekly Report", layout="wide")

ensure_default_user(cfg)
require_login()

st.subheader("Weekly Report")


def fetch_jobs(params=None) -> list:
    response = requests.get(
        f"{API_URL}/jobs",
        headers={"x-api-key": API_KEY},
        params=params,
        timeout=15,
    )
    response.raise_for_status()

    payload = response.json()

    if isinstance(payload, dict) and "items" in payload:
        return payload["items"]

    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]

    if isinstance(payload, list):
        return payload

    raise ValueError("Unexpected API response format")


def normalize_jobs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    df = frame.copy()

    for col in ["amount", "waiting_hours", "waiting_amount", "expenses_amount", "add_pay"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "job_status" not in df.columns:
        df["job_status"] = "Start"

    if "job_outcome" not in df.columns:
        df["job_outcome"] = ""

    if "paid_date" not in df.columns:
        df["paid_date"] = pd.NaT

    df["job_status"] = (
        df["job_status"]
        .fillna("Start")
        .astype(str)
        .str.strip()
        .replace("", "Start")
        .str.title()
    )

    df["job_outcome"] = (
        df["job_outcome"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.title()
    )

    if "work_date" in df.columns:
        df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date

    df["paid_date"] = pd.to_datetime(df["paid_date"], errors="coerce")

    return df


def actual_paid_received_in_week(df: pd.DataFrame, week_start: date) -> float:
    out = df.copy()

    if out.empty or "paid_date" not in out.columns or "job_status" not in out.columns:
        return 0.0

    out["job_status"] = out["job_status"].fillna("").astype(str).str.strip().str.lower()
    out["paid_date"] = pd.to_datetime(out["paid_date"], errors="coerce")

    week_start_ts = pd.to_datetime(week_start)
    week_end_ts = week_start_ts + pd.Timedelta(days=6)

    paid_df = out[
        (out["job_status"] == "paid")
        & (out["paid_date"] >= week_start_ts)
        & (out["paid_date"] <= week_end_ts)
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


today = date.today()
current_week_start = today - timedelta(days=today.weekday())

try:
    records = fetch_jobs({"page_size": 1000})
    df = pd.DataFrame(records)
    df = normalize_jobs(df)

    if df.empty:
        st.info("No jobs found.")
        st.write(f"Week starting: {current_week_start.isoformat()}")
        st.stop()

    if "work_date" not in df.columns:
        st.error("API response does not include 'work_date'.")
        st.write(df)
        st.stop()

    df = df.dropna(subset=["work_date"])

    if df.empty:
        st.info("No valid dated jobs found.")
        st.stop()

    df["_week_start"] = df["work_date"].apply(lambda d: d - timedelta(days=d.weekday()))

    all_weeks = sorted(df["_week_start"].unique().tolist(), reverse=True)

    if current_week_start in all_weeks:
        options = all_weeks
        default_index = all_weeks.index(current_week_start)
    else:
        options = [current_week_start] + all_weeks
        default_index = 0

    selected = st.selectbox(
        "Select week (Mon–Sun)",
        options,
        index=default_index,
        format_func=format_week_range,
    )

    sub = df[df["_week_start"] == selected].copy()

    if sub.empty:
        st.info(f"No jobs found for {format_week_range(selected)}.")
        st.stop()

    st.caption(f"Showing jobs for: **{format_week_range(selected)}**")

    t = compute_totals(sub)
    actual_paid = actual_paid_received_in_week(sub, selected)

except Exception as e:
    st.error(f"Failed to load jobs from API: {e}")
    st.stop()

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

display_jobs_table(cfg, sub, caption=f"Jobs for {format_week_range(selected)}", show_paid_date=True)