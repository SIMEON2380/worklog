import math
from datetime import date

import pandas as pd
import streamlit as st

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Target Intelligence", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.title("🎯 Target Intelligence")

TARGET = 200.0
today = date.today()

df = DB["read_all"]()

if df.empty:
    st.info("No worklog data found yet.")
    st.stop()

df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date

today_df = df[df["work_date"] == today].copy()

if today_df.empty:
    st.info("No jobs recorded for today yet.")
    st.stop()

for col in ["amount", "expenses_amount", "waiting_amount", "add_pay"]:
    if col in today_df.columns:
        today_df[col] = pd.to_numeric(today_df[col], errors="coerce").fillna(0.0)
    else:
        today_df[col] = 0.0

cars_driven = len(today_df)
total_amount = float(today_df["amount"].sum())
total_expenses = float(today_df["expenses_amount"].sum())
total_waiting_pay = float(today_df["waiting_amount"].sum())
total_add_pay = float(today_df["add_pay"].sum())

gross_today = total_amount + total_waiting_pay + total_add_pay
net_today = gross_today - total_expenses
gap = TARGET - net_today

avg_per_car = gross_today / cars_driven if cars_driven > 0 else 0.0
cars_needed = math.ceil(gap / avg_per_car) if gap > 0 and avg_per_car > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Cars Driven Today", cars_driven)
col2.metric("Net Today", f"£{net_today:.2f}")
col3.metric("Gap to £200", f"£{gap:.2f}" if gap > 0 else "£0.00")

col4, col5, col6 = st.columns(3)
col4.metric("Job Amount", f"£{total_amount:.2f}")
col5.metric("Waiting Pay", f"£{total_waiting_pay:.2f}")
col6.metric("Expenses", f"£{total_expenses:.2f}")

st.metric("Add Pay", f"£{total_add_pay:.2f}")
st.write(f"**Average earned per car:** £{avg_per_car:.2f}")

if gap > 0:
    st.warning(f"You need about {cars_needed} more average car(s) to hit £{TARGET:.0f}.")
else:
    st.success("Target hit. Nice.")

st.subheader("This Week Overview")

df["week_start"] = pd.to_datetime(df["work_date"], errors="coerce")
df = df.dropna(subset=["week_start"]).copy()
df["week_start"] = df["week_start"].dt.date

from datetime import timedelta
start_of_week = today - timedelta(days=today.weekday())

week_df = df[df["week_start"] >= start_of_week].copy()

for col in ["amount", "expenses_amount", "waiting_amount", "add_pay"]:
    if col in week_df.columns:
        week_df[col] = pd.to_numeric(week_df[col], errors="coerce").fillna(0.0)
    else:
        week_df[col] = 0.0

weekly_cars = len(week_df)
weekly_amount = float(week_df["amount"].sum())
weekly_waiting = float(week_df["waiting_amount"].sum())
weekly_add_pay = float(week_df["add_pay"].sum())
weekly_expenses = float(week_df["expenses_amount"].sum())
weekly_net = weekly_amount + weekly_waiting + weekly_add_pay - weekly_expenses

col7, col8, col9 = st.columns(3)
col7.metric("Cars Driven This Week", weekly_cars)
col8.metric("Weekly Net", f"£{weekly_net:.2f}")
col9.metric("Weekly Expenses", f"£{weekly_expenses:.2f}")