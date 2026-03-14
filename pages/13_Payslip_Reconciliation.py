import re
from io import BytesIO

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

# -------------------------
# Payslip Summary + Insights
# -------------------------
st.markdown("### Payslip Summary by Job")
if summary_df.empty:
    st.warning("No job lines were found in the payslip.")
    if show_debug:
        st.info(
            "Check the debug text above. If the payslip rows are visible there, "
            "the parser regex needs adjusting to the actual PDF layout."
        )
else:
    # Make sure numeric fields are numeric
    for col in [
        "job_amount",
        "expenses",
        "waiting_time",
        "regional_waiting",
        "addpay",
        "total_paid",
    ]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").fillna(0.0)

    other_total = 0.0
    if not other_df.empty and "amount" in other_df.columns:
        other_df["amount"] = pd.to_numeric(other_df["amount"], errors="coerce").fillna(0.0)
        other_total = float(other_df["amount"].sum())

    payslip_total = float(summary_df["total_paid"].sum())
    grand_total = payslip_total + other_total

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs in payslip", int(summary_df["job_id"].nunique()))
    c2.metric("Payslip total", f"£{payslip_total:.2f}")
    c3.metric("Other items total", f"£{other_total:.2f}")
    c4.metric("Grand total", f"£{grand_total:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Job pay total", f"£{summary_df['job_amount'].sum():.2f}")
    c6.metric("Expenses total", f"£{summary_df['expenses'].sum():.2f}")
    c7.metric("Waiting total", f"£{summary_df['waiting_time'].sum():.2f}")
    c8.metric("Regional waiting", f"£{summary_df['regional_waiting'].sum():.2f}")

    # Payslip Insights
    st.markdown("### Payslip Insights")

    highest_paid_job_id = "-"
    highest_paid_job_value = 0.0
    zero_paid_jobs_count = 0
    addpay_total = 0.0

    if not summary_df.empty:
        highest_paid_row = summary_df.sort_values("total_paid", ascending=False).iloc[0]
        highest_paid_job_id = str(highest_paid_row["job_id"])
        highest_paid_job_value = float(highest_paid_row["total_paid"])
        zero_paid_jobs_count = int((summary_df["job_amount"] == 0).sum())
        addpay_total = float(summary_df["addpay"].sum())

    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("Highest paid job", highest_paid_job_id)
    i2.metric("Highest job total", f"£{highest_paid_job_value:.2f}")
    i3.metric("Waiting paid", f"£{summary_df['waiting_time'].sum():.2f}")
    i4.metric("Addpay total", f"£{addpay_total:.2f}")
    i5.metric("Jobs paid £0", zero_paid_jobs_count)

    st.dataframe(summary_df, use_container_width=True)

    # Export summary to Excel
    summary_output = BytesIO()
    with pd.ExcelWriter(summary_output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Payslip Summary")
        if not other_df.empty:
            other_df.to_excel(writer, index=False, sheet_name="Other Payslip Items")
    summary_output.seek(0)

    st.download_button(
        label="Download payslip summary (Excel)",
        data=summary_output.getvalue(),
        file_name="payslip_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

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

# -------------------------
# Export reconciliation
# -------------------------
st.markdown("### Export Reconciliation")

csv_data = recon_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download reconciliation report (CSV)",
    data=csv_data,
    file_name="payslip_reconciliation.csv",
    mime="text/csv",
)

excel_output = BytesIO()
with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
    recon_df.to_excel(writer, index=False, sheet_name="Reconciliation")
    if not summary_df.empty:
        summary_df.to_excel(writer, index=False, sheet_name="Payslip Summary")
    if not other_df.empty:
        other_df.to_excel(writer, index=False, sheet_name="Other Payslip Items")
excel_output.seek(0)

st.download_button(
    label="Download reconciliation report (Excel)",
    data=excel_output.getvalue(),
    file_name="payslip_reconciliation.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)