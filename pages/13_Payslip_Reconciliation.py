import re
import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login
from worklog.payslip_parser import (
    extract_text_from_pdf,
    parse_uploaded_payslip,
    build_db_reconciliation,
)

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

show_debug = st.checkbox("Show debug output", value=True)

# -------------------------
# Debug extracted text
# -------------------------
try:
    extracted_text = extract_text_from_pdf(uploaded_file)
except Exception as e:
    st.error(f"Failed to extract text from PDF: {e}")
    st.stop()

if not extracted_text or not extracted_text.strip():
    st.warning("No text could be extracted from this PDF.")
    st.stop()

# Clean text a bit so you can compare raw vs parser-friendly version
cleaned_text = re.sub(r"[ \t]+", " ", extracted_text)
cleaned_text = re.sub(r"\r", "\n", cleaned_text)
cleaned_text = re.sub(r"\n+", "\n", cleaned_text).strip()

preview_lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]

if show_debug:
    st.markdown("### Debug Extracted Text")
    st.text_area(
        "Raw PDF text preview",
        extracted_text[:5000],
        height=250,
    )

    st.markdown("### Debug Cleaned Text")
    st.text_area(
        "Cleaned text preview",
        cleaned_text[:5000],
        height=250,
    )

    st.markdown("### Debug First Extracted Lines")
    if preview_lines:
        debug_lines_df = pd.DataFrame(
            {
                "line_no": range(1, min(len(preview_lines), 40) + 1),
                "text": preview_lines[:40],
            }
        )
        st.dataframe(debug_lines_df, use_container_width=True)
    else:
        st.info("No extracted lines to preview.")

# Reset file pointer before parsing again
uploaded_file.seek(0)

try:
    jobs_df, other_df, summary_df = parse_uploaded_payslip(uploaded_file)
except Exception as e:
    st.error(f"Failed to parse payslip PDF: {e}")
    st.stop()

st.markdown("### Payslip Summary by Job")
if summary_df.empty:
    st.warning("No job lines were found in the payslip.")
    if show_debug:
        st.info(
            "Check the debug text above. If the payslip rows are visible there, "
            "the parser regex needs adjusting to the actual PDF layout."
        )
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

# -------------------------
# Database reconciliation
# -------------------------
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

if "expenses_amount" not in db_df.columns:
    if "expenses" in db_df.columns:
        db_df["expenses_amount"] = pd.to_numeric(
            db_df["expenses"], errors="coerce"
        ).fillna(0.0)
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