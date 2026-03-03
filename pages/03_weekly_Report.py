
import streamlit as st
import pandas as pd
from datetime import date, timedelta

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, show_totals, display_jobs_table

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Weekly Report", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Weekly Report")

df = DB["read_all"]()
today = date.today()
current_week_start = today - timedelta(days=today.weekday())

if df.empty:
    st.info("No jobs found.")
    st.write(f"Week starting: {current_week_start.isoformat()}")
    st.stop()

df = df.copy()
df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date
df = df.dropna(subset=["work_date"])

df["_week_start"] = df["work_date"].apply(lambda d: d - timedelta(days=d.weekday()))

all_weeks = sorted(df["_week_start"].unique().tolist(), reverse=True)
options = [current_week_start] + [w for w in all_weeks if w != current_week_start]

selected = st.selectbox("Select week (Mon–Sun)", options, index=0)
sub = df[df["_week_start"] == selected].copy()
sub = sub.drop(columns=["_week_start"], errors="ignore")

show_totals(sub)
display_jobs_table(cfg, sub, caption="Jobs in selected week")
