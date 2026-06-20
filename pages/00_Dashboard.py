import json
import os
from datetime import date

import pandas as pd
import requests
import streamlit as st

from worklog.auth import ensure_default_user
from worklog.config import Config
from worklog.ui import require_login


API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("WORKLOG_API_KEY", "")

HEADERS = {"x-api-key": API_KEY}

cfg = Config()

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Dashboard", layout="wide")

ensure_default_user(cfg)
require_login()

st.title("🏠 Dashboard")


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
            nested_rows = extract_rows_from_payload(value)
            if nested_rows:
                return nested_rows

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


def ensure_column(df, column, default_value=""):
    if column not in df.columns:
        df[column] = default_value
    return df


def fetch_jobs():
    try:
        response = requests.get(
            f"{API_URL}/jobs",
            headers=HEADERS,
            params={"all_records": "true", "limit": 5000},
            timeout=20,
        )
    except requests.RequestException as e:
        st.error(f"API connection failed: {e}")
        return pd.DataFrame()

    if response.status_code == 401:
        st.error("API key rejected. Check WORKLOG_API_KEY.")
        return pd.DataFrame()

    if response.status_code != 200:
        st.error(f"API error: {response.status_code} - {response.text}")
        return pd.DataFrame()

    payload = response.json()
    rows = normalise_rows(extract_rows_from_payload(payload))

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    required_cols = [
        "id",
        "work_date",
        "job_id",
        "category",
        "job_status",
        "amount",
        "waiting_amount",
        "expenses_amount",
        "add_pay",
        "vehicle_reg",
        "vehicle_description",
        "collection_from",
        "delivery_to",
        "job_outcome",
    ]

    for col in required_cols:
        df = ensure_column(df, col, "")

    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce")

    money_cols = ["amount", "waiting_amount", "expenses_amount", "add_pay"]

    for col in money_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["gross_total"] = df["amount"] + df["waiting_amount"] + df["add_pay"]
    df["net_total"] = df["gross_total"] - df["expenses_amount"]

    df["job_status_clean"] = df["job_status"].astype(str).str.strip().str.lower()

    return df


df = fetch_jobs()

if df.empty:
    st.warning("No jobs found.")
    st.stop()


today = date.today()

valid_df = df.dropna(subset=["work_date"]).copy()
valid_df["report_date"] = valid_df["work_date"].dt.date

today_df = valid_df[valid_df["report_date"] == today].copy()

week_start = pd.Timestamp(today).to_period("W").start_time.date()
week_end = pd.Timestamp(today).to_period("W").end_time.date()

week_df = valid_df[
    (valid_df["report_date"] >= week_start)
    & (valid_df["report_date"] <= week_end)
].copy()

month_df = valid_df[
    valid_df["work_date"].dt.to_period("M") == pd.Timestamp(today).to_period("M")
].copy()

outstanding_statuses = ["start", "pending", "completed"]

outstanding_df = valid_df[
    valid_df["job_status_clean"].isin(outstanding_statuses)
].copy()

paid_df = valid_df[valid_df["job_status_clean"] == "paid"].copy()


today_revenue = today_df["net_total"].sum()
week_revenue = week_df["net_total"].sum()
month_revenue = month_df["net_total"].sum()
outstanding_amount = outstanding_df["net_total"].sum()

st.markdown("### 📊 Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Today Revenue", f"£{today_revenue:,.2f}")
col2.metric("This Week Revenue", f"£{week_revenue:,.2f}")
col3.metric("This Month Revenue", f"£{month_revenue:,.2f}")
col4.metric("Outstanding", f"£{outstanding_amount:,.2f}")

st.markdown("### 🚗 Job Activity")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Jobs Today", len(today_df))
col2.metric("Jobs This Week", len(week_df))
col3.metric("Jobs This Month", len(month_df))
col4.metric("Paid Jobs", len(paid_df))

st.markdown("### 💰 Financial Breakdown")

breakdown_col1, breakdown_col2, breakdown_col3, breakdown_col4 = st.columns(4)

total_amount = valid_df["amount"].sum()
total_waiting = valid_df["waiting_amount"].sum()
total_expenses = valid_df["expenses_amount"].sum()
total_net = valid_df["net_total"].sum()

breakdown_col1.metric("Revenue", f"£{total_amount:,.2f}")
breakdown_col2.metric("Waiting Pay", f"£{total_waiting:,.2f}")
breakdown_col3.metric("Expenses", f"£{total_expenses:,.2f}")
breakdown_col4.metric("Net Earnings", f"£{total_net:,.2f}")

st.divider()

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("### 🔴 Outstanding Jobs")

    if outstanding_df.empty:
        st.success("No outstanding jobs.")
    else:
        outstanding_cols = [
            "work_date",
            "job_id",
            "job_status",
            "amount",
            "waiting_amount",
            "add_pay",
            "expenses_amount",
            "net_total",
            "vehicle_reg",
            "collection_from",
            "delivery_to",
        ]

        available_cols = [col for col in outstanding_cols if col in outstanding_df.columns]

        st.dataframe(
            outstanding_df[available_cols]
            .sort_values("work_date", ascending=False)
            .head(50),
            use_container_width=True,
        )

with right_col:
    st.markdown("### 🕘 Recent Jobs")

    recent_cols = [
        "work_date",
        "job_id",
        "vehicle_reg",
        "job_status",
        "amount",
        "net_total",
    ]

    available_recent_cols = [col for col in recent_cols if col in valid_df.columns]

    st.dataframe(
        valid_df[available_recent_cols]
        .sort_values("work_date", ascending=False)
        .head(10),
        use_container_width=True,
    )

st.divider()

st.markdown("### 📈 Quick Insights")

insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)

average_job_value = valid_df["net_total"].mean() if not valid_df.empty else 0
average_expenses = valid_df["expenses_amount"].mean() if not valid_df.empty else 0

highest_job = valid_df.sort_values("net_total", ascending=False).head(1)

if not highest_job.empty:
    highest_job_number = str(highest_job.iloc[0].get("job_id") or "")
    highest_job_value = float(highest_job.iloc[0].get("net_total") or 0)
else:
    highest_job_number = "N/A"
    highest_job_value = 0

if "vehicle_description" in valid_df.columns and not valid_df.empty:
    vehicle_summary = (
        valid_df.groupby("vehicle_description", dropna=False)
        .agg(total_net=("net_total", "sum"))
        .reset_index()
        .sort_values("total_net", ascending=False)
    )

    top_vehicle = vehicle_summary.iloc[0]["vehicle_description"] if not vehicle_summary.empty else "N/A"
    top_vehicle_value = vehicle_summary.iloc[0]["total_net"] if not vehicle_summary.empty else 0
else:
    top_vehicle = "N/A"
    top_vehicle_value = 0

insight_col1.metric("Average Job Value", f"£{average_job_value:,.2f}")
insight_col2.metric("Average Expenses", f"£{average_expenses:,.2f}")
insight_col3.metric("Highest Job", f"£{highest_job_value:,.2f}", highest_job_number)
insight_col4.metric("Top Vehicle", f"£{top_vehicle_value:,.2f}", str(top_vehicle))

st.divider()

st.markdown("### 📅 Monthly Revenue")

monthly_chart_df = valid_df.copy()
monthly_chart_df["month"] = monthly_chart_df["work_date"].dt.to_period("M").astype(str)

monthly_revenue = (
    monthly_chart_df.groupby("month")
    .agg(net_total=("net_total", "sum"))
    .reset_index()
    .sort_values("month")
)

if not monthly_revenue.empty:
    st.bar_chart(monthly_revenue, x="month", y="net_total")
else:
    st.info("No monthly revenue data available.")

st.markdown("### 📆 Weekly Revenue")

weekly_chart_df = valid_df.copy()
weekly_chart_df["week"] = weekly_chart_df["work_date"].dt.to_period("W").astype(str)

weekly_revenue = (
    weekly_chart_df.groupby("week")
    .agg(net_total=("net_total", "sum"))
    .reset_index()
    .sort_values("week")
    .tail(12)
)

if not weekly_revenue.empty:
    st.bar_chart(weekly_revenue, x="week", y="net_total")
else:
    st.info("No weekly revenue data available.")