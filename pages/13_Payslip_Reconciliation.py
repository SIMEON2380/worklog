import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login
from worklog.payslip_parser import parse_uploaded_payslip, build_db_reconciliation

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Payslip Reconciliation", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Payslip Reconciliation")

uploaded_file = st.file_uploader("Upload payslip PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a payslip PDF to begin.")
    st.stop()

try:
    jobs_df, other_df, summary_df = parse_uploaded_payslip(uploaded_file)
except Exception as e:
    st.error(f"Failed to parse payslip PDF: {e}")
    st.stop()

st.markdown("### Payslip Summary by Job")
if summary_df.empty:
    st.warning("No job lines were found in the payslip.")
else:
    st.dataframe(summary_df, use_container_width=True)

st.markdown("### Raw Job Lines")
if jobs_df.empty:
    st.info("No raw job lines found.")
else:
    st.dataframe(jobs_df, use_container_width=True)

st.markdown("### Other Payslip Items")
if other_df.empty:
    st.info("No other items found.")
else:
    st.dataframe(other_df, use_container_width=True)

# Database reconciliation
db_df = DB["read_all"]()

st.markdown("### Reconciliation Result")

if summary_df.empty:
    st.info("No payslip jobs available for reconciliation.")
    st.stop()

if db_df is None or db_df.empty:
    st.warning("Database has no jobs to compare against.")
    st.stop()

db_df = db_df.copy()
db_df["job_id"] = db_df["job_id"].astype(str).str.strip()

# Keep this simple and safe.
# Main amount comes from DB["amount"].
# Expenses use expenses_amount if present, otherwise fallback to expenses if numeric.
# Waiting time stays 0 unless you already store a money column for it.

if "expenses_amount" not in db_df.columns:
    if "expenses" in db_df.columns:
        db_df["expenses_amount"] = pd.to_numeric(db_df["expenses"], errors="coerce").fillna(0.0)
    else:
        db_df["expenses_amount"] = 0.0

if "waiting_time_amount" not in db_df.columns:
    db_df["waiting_time_amount"] = 0.0

recon_df = build_db_reconciliation(summary_df, db_df)

if recon_df.empty:
    st.info("No reconciliation data available.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Matched jobs", int((recon_df["status"] == "Matched").sum()))
c2.metric("Issues found", int((recon_df["status"] != "Matched").sum()))
c3.metric("Net difference", f"£{recon_df['difference'].sum():.2f}")

show_only_issues = st.checkbox("Show only mismatches", value=True)

display_df = recon_df.copy()
if show_only_issues:
    display_df = display_df[display_df["status"] != "Matched"].copy()

st.dataframe(display_df, use_container_width=True)