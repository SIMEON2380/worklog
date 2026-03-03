import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, show_totals, display_jobs_table, editable_jobs_table

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Job Status", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Job Status")

df = DB["read_all"]()
if df.empty:
    st.info("No jobs found.")
    st.stop()

df = df.copy()
df["job_status"] = df["job_status"].fillna("Pending").astype(str)

status = st.selectbox("Select status", options=cfg.STATUS_OPTIONS, index=0)

sub = df[df["job_status"] == status].copy()

show_totals(sub)

st.caption(f"Jobs with status: {status} (editable)")
editable_jobs_table(
    cfg=cfg,
    DB=DB,
    df_db=sub,
    key=f"status_editor_{status}",
    allow_type_edit=True,
)