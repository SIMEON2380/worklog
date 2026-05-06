import json
import os
from datetime import date

import pandas as pd
import requests
import streamlit as st

from worklog.auth import ensure_default_user
from worklog.config import Config
from worklog.reporting import compute_totals
from worklog.ui import require_login, display_jobs_table


API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("WORKLOG_API_KEY", "")

cfg = Config()

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Daily Report", layout="wide")

ensure_default_user(cfg)
require_login()

st.subheader("Daily Report")


def parse_json_if_needed(value):
    if isinstance(value, str):
        value = value.strip()

        if value.startswith("{") and value.endswith("}"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

    return value


def extract_rows_from_payload(payload):
    payload = parse_json_if_needed(payload)

    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    for key in ["data", "items", "results", "rows"]:
        value = parse_json_if_needed(payload.get(key))

        if isinstance(value, list):
            return value

        if isinstance(value, dict):
            nested = extract_rows_from_payload(value)
            if nested:
                return nested

    return []


def normalise_rows(rows):
    normalised = []

    for row in rows:
        row = parse_json_if_needed(row)

        if isinstance(row, dict) and "data" in row:
            inner = parse_json_if_needed(row.get("data"))

            if isinstance(inner, dict):
                normalised.append(inner)
                continue

        if isinstance(row, dict):
            normalised.append(row)

    return normalised


def normalize_jobs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    df = frame.copy()

    if "work_date" not in df.columns and "data" in df.columns:
        parsed_rows = normalise_rows(df["data"].tolist())
        df = pd.DataFrame(parsed_rows)

    for col in [
        "amount",
        "waiting_hours",
        "waiting_amount",
        "expenses_amount",
        "add_pay",
    ]:
        if col not in df.columns:
            df[col] = 0.0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "job_status" not in df.columns:
        df["job_status"] = "Start"

    if "job_outcome" not in df.columns:
        df["job_outcome"] = ""

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

    return df


try:
    response = requests.get(
        f"{API_URL}/jobs",
        headers={"x-api-key": API_KEY},
        params={"all_records": "true", "limit": 5000},
        timeout=20,
    )

    response.raise_for_status()

    payload = response.json()
    records = extract_rows_from_payload(payload)
    records = normalise_rows(records)

    df = pd.DataFrame(records)
    df = normalize_jobs(df)

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