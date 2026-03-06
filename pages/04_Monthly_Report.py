import streamlit as st
import pandas as pd
from datetime import date

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, display_jobs_table

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Monthly Report", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Monthly Report")

df = DB["read_all"]()
today = date.today()
current_month = today.strftime("%Y-%m")

if df.empty:
    st.info("No jobs found.")
    st.write(f"Month: {current_month}")
    st.stop()

df = df.copy()
df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce")
df = df.dropna(subset=["work_date"])
df["_month"] = df["work_date"].dt.to_period("M").astype(str)

all_months = sorted(df["_month"].unique().tolist(), reverse=True)
options = [current_month] + [m for m in all_months if m != current_month]

selected = st.selectbox("Select month", options, index=0)

sub = df[df["_month"] == selected].copy()

st.write(f"Month: {selected}")

if sub.empty:
    st.info("No jobs found for this month.")
    st.stop()

display_jobs_table(sub.drop(columns=["_month"], errors="ignore"))