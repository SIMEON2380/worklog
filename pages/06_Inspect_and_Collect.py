import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, show_totals, display_jobs_table, editable_jobs_table

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

# Your db column for job type is `category`
df = df.copy()
df["category"] = df["category"].fillna("").astype(str)

# Filter to Inspect & Collect types
sub = df[df["category"].isin(list(cfg.INSPECT_COLLECT_TYPES))].copy()

show_totals(sub)

st.caption("Inspect & Collect jobs (editable)")
editable_jobs_table(
    cfg=cfg,
    DB=DB,
    df_db=sub,
    key="inspect_collect_editor",
    allow_type_edit=False,   # keeps job type locked; you can change status to Paid
)