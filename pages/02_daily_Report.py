
import streamlit as st
import pandas as pd
from datetime import date

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, show_totals, display_jobs_table

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Daily Report", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Daily Report")

df = DB["read_all"]()
today = date.today()

if df.empty:
    st.info("No jobs found.")
    st.write(f"Date: {today.isoformat()}")
    st.stop()

df = df.copy()
df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date
df = df.dropna(subset=["work_date"])

all_days = sorted(df["work_date"].unique().tolist(), reverse=True)
options = [today] + [d for d in all_days if d != today]

selected = st.selectbox("Select day", options, index=0)
sub = df[df["work_date"] == selected].copy()

show_totals(sub)
display_jobs_table(cfg, sub, caption="Jobs in selected day")
