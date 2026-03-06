import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, editable_jobs_table

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Inspect & Collect", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Inspect & Collect")

df = DB["read_all"]()
if df.empty:
    st.info("No jobs found.")
    st.stop()

df = df.copy()
df["category"] = df["category"].fillna("").astype(str)

# Only Inspect & Collect job types
sub = df[df["category"].isin(list(cfg.INSPECT_COLLECT_TYPES))].copy()

if sub.empty:
    st.info("No Inspect & Collect jobs found.")
    st.stop()

# -------------------------
# Vehicle Reg filter only
# -------------------------
if "vehicle_reg" not in sub.columns:
    st.error("Missing 'vehicle_reg' column in dataset.")
    st.stop()

reg_search = st.text_input("Filter by Vehicle Reg")

if reg_search:
    sub = sub[
        sub["vehicle_reg"].astype(str).str.contains(reg_search, case=False, na=False)
    ].copy()

if sub.empty:
    st.info("No Inspect & Collect jobs match that vehicle reg.")
    st.stop()

# Make sure numbers are numeric
sub["amount"] = pd.to_numeric(sub.get("amount", 0), errors="coerce").fillna(0.0)
sub["waiting_amount"] = pd.to_numeric(sub.get("waiting_amount", 0), errors="coerce").fillna(0.0)
sub["waiting_hours"] = pd.to_numeric(sub.get("waiting_hours", 0), errors="coerce").fillna(0.0)

# Make sure optional display columns exist
if "waiting_time" not in sub.columns:
    sub["waiting_time"] = ""

if "job_status" not in sub.columns:
    sub["job_status"] = ""

# Per-job computed pay
sub["inspect_base"] = float(cfg.INSPECT_COLLECT_RATE)  # always £8
sub["job_total"] = sub["inspect_base"] + sub["waiting_amount"] + sub["amount"]

# Totals
total_jobs = int(len(sub))
total_base = float(sub["inspect_base"].sum())
total_wait_pay = float(sub["waiting_amount"].sum())
total_wait_hours = float(sub["waiting_hours"].sum())
total_job_amount = float(sub["amount"].sum())
grand_total = float(sub["job_total"].sum())

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Jobs", f"{total_jobs}")
c2.metric("Base (£8 each)", f"£{total_base:,.2f}")
c3.metric("Waiting Hours", f"{total_wait_hours:,.2f} hrs")
c4.metric("Waiting Pay", f"£{total_wait_pay:,.2f}")
c5.metric("Job Amount", f"£{total_job_amount:,.2f}")
c6.metric("Grand Total", f"£{grand_total:,.2f}")

st.divider()

st.caption("Inspect & Collect pay breakdown (computed)")
breakdown = sub[[
    "work_date",
    "job_id",
    "vehicle_reg",
    "category",
    "job_status",
    "inspect_base",
    "waiting_time",
    "waiting_hours",
    "waiting_amount",
    "amount",
    "job_total",
]].copy()

st.dataframe(breakdown, use_container_width=True)

st.divider()

st.caption("Edit Inspect & Collect jobs (you can change status to Paid, and waiting/job amount if needed)")
editable_jobs_table(
    cfg=cfg,
    DB=DB,
    df_db=sub,
    key="inspect_collect_editor",
    allow_type_edit=False,
)