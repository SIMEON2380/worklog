import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login
from worklog.payslip_parser import parse_uploaded_payslip

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Payslip Reconciliation", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Payslip Reconciliation")

uploaded_file = st.file_uploader("Upload payslip PDF", type=["pdf"])

if uploaded_file is not None:
    jobs_df, other_df, summary_df = parse_uploaded_payslip(uploaded_file)

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