import os

import pandas as pd
import requests
import streamlit as st

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, editable_jobs_table

API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("WORKLOG_API_KEY", "supersecret123")

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Job Status", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Job Status Dashboard")


def safe_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def get_status_col(frame: pd.DataFrame) -> str | None:
    if "job_status" in frame.columns:
        return "job_status"
    if "status" in frame.columns:
        return "status"
    return None


def compute_pending_money(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0

    out = frame.copy()

    out["amount"] = safe_num(out["amount"]) if "amount" in out.columns else 0.0
    out["waiting_amount"] = safe_num(out["waiting_amount"]) if "waiting_amount" in out.columns else 0.0
    out["add_pay"] = safe_num(out["add_pay"]) if "add_pay" in out.columns else 0.0
    out["expenses_amount"] = safe_num(out["expenses_amount"]) if "expenses_amount" in out.columns else 0.0

    total = (
        float(out["amount"].sum())
        + float(out["waiting_amount"].sum())
        + float(out["add_pay"].sum())
        - float(out["expenses_amount"].sum())
    )
    return round(total, 2)


def format_date_column(frame: pd.DataFrame, col_name: str = "work_date") -> pd.DataFrame:
    temp = frame.copy()
    if col_name in temp.columns:
        temp[col_name] = pd.to_datetime(temp[col_name], errors="coerce").dt.strftime("%Y-%m-%d")
    return temp


def fetch_jobs(params: dict) -> tuple[pd.DataFrame, dict]:
    response = requests.get(
        f"{API_URL}/jobs",
        headers={"x-api-key": API_KEY},
        params=params,
        timeout=20,
    )

    if response.status_code != 200:
        raise RuntimeError(f"API failed: {response.status_code} | {response.text}")

    payload = response.json()

    if isinstance(payload, dict) and "data" in payload:
        records = payload.get("data", [])
        meta = {
            "page": int(payload.get("page", 1)),
            "page_size": int(payload.get("page_size", len(records) if records else 50)),
            "total": int(payload.get("total", len(records))),
            "total_pages": int(payload.get("total_pages", 1)),
        }
    elif isinstance(payload, list):
        records = payload
        meta = {
            "page": 1,
            "page_size": len(records) if records else 50,
            "total": len(records),
            "total_pages": 1,
        }
    else:
        raise RuntimeError(f"Unexpected API response format: {payload}")

    return pd.DataFrame(records).copy(), meta


status_options = ["All", "Paid", "Pending", "Start", "Completed", "Aborted", "Withdraw"]
outcome_options = ["All", "Completed", "Aborted", "Withdraw", "Fail"]
type_options = ["All"] + list(getattr(cfg, "JOB_TYPE_OPTIONS", []))
page_size_options = [25, 50, 100, 200]

if "job_status_page_num" not in st.session_state:
    st.session_state["job_status_page_num"] = 1

f1, f2, f3, f4 = st.columns(4)

selected_api_status = f1.selectbox("Payment Status", options=status_options, index=0)
selected_outcome = f2.selectbox("Job Outcome", options=outcome_options, index=0)
selected_type = f3.selectbox("Job Type", options=type_options, index=0)
search_text = f4.text_input("Search Job Number / Vehicle Reg / Vehicle")

d1, d2, d3 = st.columns(3)
selected_start_date = d1.date_input("Start Date", value=None)
selected_end_date = d2.date_input("End Date", value=None)
page_size = d3.selectbox("Rows Per Page", options=page_size_options, index=1)

params = {
    "page": int(st.session_state["job_status_page_num"]),
    "page_size": int(page_size),
}

if selected_api_status != "All":
    params["job_status"] = selected_api_status

if selected_outcome != "All":
    params["job_outcome"] = selected_outcome

if selected_type != "All":
    params["category"] = selected_type

if search_text.strip():
    params["search"] = search_text.strip()

if selected_start_date is not None:
    params["start_date"] = pd.to_datetime(selected_start_date).strftime("%Y-%m-%d")

if selected_end_date is not None:
    params["end_date"] = pd.to_datetime(selected_end_date).strftime("%Y-%m-%d")

try:
    df, meta = fetch_jobs(params)
except Exception as e:
    st.error(f"Failed to load jobs from API: {e}")
    st.stop()

total_rows = int(meta.get("total", 0))
current_page_value = int(meta.get("page", 1))
page_size_value = int(meta.get("page_size", 50))
total_pages = max(int(meta.get("total_pages", 1)), 1)

if st.session_state["job_status_page_num"] > total_pages:
    st.session_state["job_status_page_num"] = total_pages
    st.rerun()

if df.empty:
    st.info("No jobs found for the selected filters.")
    st.stop()

status_col = get_status_col(df)
if status_col is None:
    st.error("No status column found in the data.")
    st.stop()

df[status_col] = safe_text(df[status_col])
df["job_outcome"] = safe_text(df["job_outcome"]) if "job_outcome" in df.columns else ""
df["category"] = safe_text(df["category"]) if "category" in df.columns else ""
df["vehicle_reg"] = safe_text(df["vehicle_reg"]) if "vehicle_reg" in df.columns else ""
df["job_id"] = safe_text(df["job_id"]) if "job_id" in df.columns else ""
df["vehicle_description"] = safe_text(df["vehicle_description"]) if "vehicle_description" in df.columns else ""

if "work_date" in df.columns:
    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce")

if "id" not in df.columns:
    df["id"] = range(1, len(df) + 1)

df["amount"] = safe_num(df["amount"]) if "amount" in df.columns else 0.0
df["waiting_amount"] = safe_num(df["waiting_amount"]) if "waiting_amount" in df.columns else 0.0
df["add_pay"] = safe_num(df["add_pay"]) if "add_pay" in df.columns else 0.0
df["expenses_amount"] = safe_num(df["expenses_amount"]) if "expenses_amount" in df.columns else 0.0
df["gross_value"] = df["amount"] + df["waiting_amount"] + df["add_pay"] - df["expenses_amount"]

start_row = ((current_page_value - 1) * page_size_value) + 1 if total_rows > 0 else 0
end_row = min(current_page_value * page_size_value, total_rows)

st.caption(
    f"Showing {start_row}-{end_row} of {total_rows} jobs "
    f"(page {current_page_value} of {total_pages})"
)

nav1, nav2, nav3 = st.columns([1, 1, 2])

with nav1:
    if st.button("Previous Page", disabled=current_page_value <= 1):
        st.session_state["job_status_page_num"] = max(1, current_page_value - 1)
        st.rerun()

with nav2:
    if st.button("Next Page", disabled=current_page_value >= total_pages):
        st.session_state["job_status_page_num"] = min(total_pages, current_page_value + 1)
        st.rerun()

with nav3:
    st.write(f"Current page: {current_page_value}")

st.divider()

paid_df = df[df[status_col].str.lower() == "paid"].copy()
pending_df = df[df[status_col].str.lower() == "pending"].copy()
aborted_df = df[df["job_outcome"].str.lower() == "aborted"].copy()
withdraw_df = df[df["job_outcome"].str.lower() == "withdraw"].copy()
fail_df = df[df["job_outcome"].str.lower() == "fail"].copy()
aborted_paid_df = df[
    (df["job_outcome"].str.lower() == "aborted")
    & (df[status_col].str.lower() == "paid")
].copy()

pending_money = compute_pending_money(pending_df)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Page Jobs", int(len(df)))
k2.metric("Paid", int(len(paid_df)))
k3.metric("Pending", int(len(pending_df)))
k4.metric("Aborted", int(len(aborted_df)))
k5.metric("Withdraw", int(len(withdraw_df)))
k6.metric("Pending £", f"£{pending_money:,.2f}")

st.divider()

left, right = st.columns(2)

with left:
    st.markdown("### Payment Status Summary")
    payment_summary = (
        df.groupby(status_col, dropna=False)
        .agg(
            jobs=("id", "count"),
            total_value=("gross_value", "sum"),
        )
        .reset_index()
        .sort_values("jobs", ascending=False)
    )
    payment_summary["total_value"] = payment_summary["total_value"].round(2)
    st.dataframe(payment_summary, use_container_width=True, hide_index=True)

with right:
    st.markdown("### Job Outcome Summary")
    outcome_summary = (
        df.groupby("job_outcome", dropna=False)
        .agg(
            jobs=("id", "count"),
            total_value=("gross_value", "sum"),
        )
        .reset_index()
        .sort_values("jobs", ascending=False)
    )
    outcome_summary["total_value"] = outcome_summary["total_value"].round(2)
    st.dataframe(outcome_summary, use_container_width=True, hide_index=True)

st.divider()

a1, a2 = st.columns(2)

with a1:
    st.markdown("### Pending Jobs")
    if pending_df.empty:
        st.info("No pending jobs.")
    else:
        show_cols = [
            c for c in [
                "work_date",
                "job_id",
                "vehicle_reg",
                "category",
                "job_outcome",
                status_col,
                "gross_value",
            ]
            if c in pending_df.columns
        ]
        temp = format_date_column(pending_df[show_cols].copy(), "work_date")
        if "gross_value" in temp.columns:
            temp["gross_value"] = pd.to_numeric(temp["gross_value"], errors="coerce").fillna(0).round(2)
        st.dataframe(temp, use_container_width=True, hide_index=True)

with a2:
    st.markdown("### Aborted But Paid")
    if aborted_paid_df.empty:
        st.info("No aborted paid jobs.")
    else:
        show_cols = [
            c for c in [
                "work_date",
                "job_id",
                "vehicle_reg",
                "category",
                "job_outcome",
                status_col,
                "gross_value",
            ]
            if c in aborted_paid_df.columns
        ]
        temp = format_date_column(aborted_paid_df[show_cols].copy(), "work_date")
        if "gross_value" in temp.columns:
            temp["gross_value"] = pd.to_numeric(temp["gross_value"], errors="coerce").fillna(0).round(2)
        st.dataframe(temp, use_container_width=True, hide_index=True)

b1, b2 = st.columns(2)

with b1:
    st.markdown("### Withdraw Jobs")
    if withdraw_df.empty:
        st.info("No withdraw jobs.")
    else:
        show_cols = [c for c in ["work_date", "job_id", "vehicle_reg", "category", "job_outcome", status_col] if c in withdraw_df.columns]
        temp = format_date_column(withdraw_df[show_cols].copy(), "work_date")
        st.dataframe(temp, use_container_width=True, hide_index=True)

with b2:
    st.markdown("### Failed Jobs")
    if fail_df.empty:
        st.info("No failed jobs.")
    else:
        show_cols = [c for c in ["work_date", "job_id", "vehicle_reg", "category", "job_outcome", status_col] if c in fail_df.columns]
        temp = format_date_column(fail_df[show_cols].copy(), "work_date")
        st.dataframe(temp, use_container_width=True, hide_index=True)

st.divider()

st.markdown("### Payments To Chase")

if "work_date" in df.columns:
    chase_df = df.copy()
    chase_df["work_date"] = pd.to_datetime(chase_df["work_date"], errors="coerce")
    today = pd.Timestamp.today().normalize()

    chase_df = chase_df[chase_df[status_col].str.lower() == "pending"].copy()
    chase_df = chase_df[chase_df["work_date"].notna()].copy()

    if not chase_df.empty:
        chase_df["days_pending"] = (today - chase_df["work_date"]).dt.days
        chase_df["days_pending"] = pd.to_numeric(chase_df["days_pending"], errors="coerce").fillna(0).astype(int)

        pending_7 = chase_df[chase_df["days_pending"] >= 7].copy()
        pending_14 = chase_df[chase_df["days_pending"] >= 14].copy()

        c1, c2 = st.columns(2)
        c1.metric("Pending 7+ Days", int(len(pending_7)))
        c2.metric("Pending 14+ Days", int(len(pending_14)))

        x1, x2 = st.columns(2)

        with x1:
            st.markdown("#### Pending 7+ Days")
            if pending_7.empty:
                st.info("No pending jobs older than 7 days.")
            else:
                show_cols = [
                    c for c in [
                        "work_date",
                        "job_id",
                        "vehicle_reg",
                        "category",
                        "job_outcome",
                        status_col,
                        "gross_value",
                        "days_pending",
                    ]
                    if c in pending_7.columns
                ]
                temp7 = pending_7[show_cols].copy().sort_values("days_pending", ascending=False)
                temp7 = format_date_column(temp7, "work_date")
                if "gross_value" in temp7.columns:
                    temp7["gross_value"] = pd.to_numeric(temp7["gross_value"], errors="coerce").fillna(0).round(2)
                st.dataframe(temp7, use_container_width=True, hide_index=True)

        with x2:
            st.markdown("#### Pending 14+ Days")
            if pending_14.empty:
                st.info("No pending jobs older than 14 days.")
            else:
                show_cols = [
                    c for c in [
                        "work_date",
                        "job_id",
                        "vehicle_reg",
                        "category",
                        "job_outcome",
                        status_col,
                        "gross_value",
                        "days_pending",
                    ]
                    if c in pending_14.columns
                ]
                temp14 = pending_14[show_cols].copy().sort_values("days_pending", ascending=False)
                temp14 = format_date_column(temp14, "work_date")
                if "gross_value" in temp14.columns:
                    temp14["gross_value"] = pd.to_numeric(temp14["gross_value"], errors="coerce").fillna(0).round(2)
                st.dataframe(temp14, use_container_width=True, hide_index=True)
    else:
        st.info("No pending jobs with valid work dates found.")
else:
    st.info("work_date column not found, so payment age tracking cannot be shown yet.")

st.divider()

p1, p2, p3 = st.columns(3)
p1.metric("Current Page", current_page_value)
p2.metric("Total Pages", total_pages)
p3.metric("Total Matching Jobs", total_rows)

st.divider()

st.markdown("### Full Filtered Jobs Table")
st.caption("This table is editable. Saving now goes through the API.")

editable_jobs_table(
    cfg=cfg,
    DB=DB,
    df_db=df,
    key=f"job_status_dashboard_editor_page_{current_page_value}",
    allow_type_edit=True,
    api_url=API_URL,
    api_key=API_KEY,
)