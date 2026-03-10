import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, editable_jobs_table

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Job Status", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Job Status Dashboard")

df = DB["read_all"]().copy()
if df.empty:
    st.info("No jobs found.")
    st.stop()


# -------------------------
# Helpers
# -------------------------
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


status_col = get_status_col(df)

if status_col is None:
    st.error("No status column found in the database.")
    st.stop()

df[status_col] = safe_text(df[status_col])
df["job_outcome"] = safe_text(df["job_outcome"]) if "job_outcome" in df.columns else "Completed"
df["category"] = safe_text(df["category"]) if "category" in df.columns else ""
df["vehicle_reg"] = safe_text(df["vehicle_reg"]) if "vehicle_reg" in df.columns else ""
df["job_id"] = safe_text(df["job_id"]) if "job_id" in df.columns else ""

if "work_date" in df.columns:
    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce")

# numeric fields for summaries
df["amount"] = safe_num(df["amount"]) if "amount" in df.columns else 0.0
df["waiting_amount"] = safe_num(df["waiting_amount"]) if "waiting_amount" in df.columns else 0.0
df["add_pay"] = safe_num(df["add_pay"]) if "add_pay" in df.columns else 0.0
df["expenses_amount"] = safe_num(df["expenses_amount"]) if "expenses_amount" in df.columns else 0.0

df["gross_value"] = df["amount"] + df["waiting_amount"] + df["add_pay"] - df["expenses_amount"]

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

# -------------------------
# KPI cards
# -------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Jobs", int(len(df)))
k2.metric("Paid", int(len(paid_df)))
k3.metric("Pending", int(len(pending_df)))
k4.metric("Aborted", int(len(aborted_df)))
k5.metric("Withdraw", int(len(withdraw_df)))
k6.metric("Pending £", f"£{pending_money:,.2f}")

st.divider()

# -------------------------
# Filters
# -------------------------
f1, f2, f3, f4 = st.columns(4)

status_options = ["All"] + sorted([x for x in df[status_col].dropna().unique().tolist() if str(x).strip() != ""])
outcome_options = ["All"] + sorted([x for x in df["job_outcome"].dropna().unique().tolist() if str(x).strip() != ""])
type_options = ["All"] + sorted([x for x in df["category"].dropna().unique().tolist() if str(x).strip() != ""])

selected_status = f1.selectbox("Payment Status", options=status_options, index=0)
selected_outcome = f2.selectbox("Job Outcome", options=outcome_options, index=0)
selected_type = f3.selectbox("Job Type", options=type_options, index=0)
search_text = f4.text_input("Search Job Number / Vehicle Reg")

sub = df.copy()

if selected_status != "All":
    sub = sub[sub[status_col] == selected_status]

if selected_outcome != "All":
    sub = sub[sub["job_outcome"] == selected_outcome]

if selected_type != "All":
    sub = sub[sub["category"] == selected_type]

if search_text.strip():
    s = search_text.strip().lower()
    mask = (
        sub["job_id"].str.lower().str.contains(s, na=False)
        | sub["vehicle_reg"].str.lower().str.contains(s, na=False)
    )
    sub = sub[mask]

st.divider()

# -------------------------
# Summary tables
# -------------------------
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

# -------------------------
# Action sections
# -------------------------
a1, a2 = st.columns(2)

with a1:
    st.markdown("### Pending Jobs")
    if pending_df.empty:
        st.info("No pending jobs.")
    else:
        show_cols = [c for c in ["work_date", "job_id", "vehicle_reg", "category", "job_outcome", status_col, "gross_value"] if c in pending_df.columns]
        temp = pending_df[show_cols].copy()
        if "work_date" in temp.columns:
            temp["work_date"] = pd.to_datetime(temp["work_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "gross_value" in temp.columns:
            temp["gross_value"] = pd.to_numeric(temp["gross_value"], errors="coerce").fillna(0).round(2)
        st.dataframe(temp, use_container_width=True, hide_index=True)

with a2:
    st.markdown("### Aborted But Paid")
    if aborted_paid_df.empty:
        st.info("No aborted paid jobs.")
    else:
        show_cols = [c for c in ["work_date", "job_id", "vehicle_reg", "category", "job_outcome", status_col, "gross_value"] if c in aborted_paid_df.columns]
        temp = aborted_paid_df[show_cols].copy()
        if "work_date" in temp.columns:
            temp["work_date"] = pd.to_datetime(temp["work_date"], errors="coerce").dt.strftime("%Y-%m-%d")
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
        temp = withdraw_df[show_cols].copy()
        if "work_date" in temp.columns:
            temp["work_date"] = pd.to_datetime(temp["work_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        st.dataframe(temp, use_container_width=True, hide_index=True)

with b2:
    st.markdown("### Failed Jobs")
    if fail_df.empty:
        st.info("No failed jobs.")
    else:
        show_cols = [c for c in ["work_date", "job_id", "vehicle_reg", "category", "job_outcome", status_col] if c in fail_df.columns]
        temp = fail_df[show_cols].copy()
        if "work_date" in temp.columns:
            temp["work_date"] = pd.to_datetime(temp["work_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        st.dataframe(temp, use_container_width=True, hide_index=True)

st.divider()

# -------------------------
# Full filtered editable table
# -------------------------
st.markdown("### Full Filtered Jobs Table")
st.caption("This table is editable.")

if sub.empty:
    st.warning("No jobs match the selected filters.")
else:
    editable_jobs_table(
        cfg=cfg,
        DB=DB,
        df_db=sub,
        key="job_status_dashboard_editor",
        allow_type_edit=True,
    )