import math
from datetime import date, timedelta

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

TARGET = cfg.DAILY_TARGET
today = date.today()

df = DB["read_all"]()

if df.empty:
    st.info("No worklog data found yet.")
    st.stop()

df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date
df = df.dropna(subset=["work_date"]).copy()

today_df = df[df["work_date"] == today].copy()

if today_df.empty:
    st.info("No jobs recorded for today yet.")
    st.stop()

for col in ["amount", "expenses_amount", "waiting_amount", "add_pay"]:
    if col in today_df.columns:
        today_df[col] = pd.to_numeric(today_df[col], errors="coerce").fillna(0.0)
    else:
        today_df[col] = 0.0

# Count only real completed driving jobs for today
if "job_outcome" in today_df.columns:
    today_cars_df = today_df[
        today_df["job_outcome"].fillna("").str.strip().str.lower() == "completed"
    ].copy()
else:
    today_cars_df = today_df.copy()

today_cars_df = today_cars_df[today_cars_df["amount"] > 0].copy()

if "category" in today_cars_df.columns:
    today_cars_df = today_cars_df[
        today_cars_df["category"].fillna("").str.strip().str.lower() == "strd trade plate"
    ].copy()

if "job_id" in today_cars_df.columns:
    cars_driven = (
        today_cars_df["job_id"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )
else:
    cars_driven = len(today_cars_df)

total_amount = float(today_df["amount"].sum())
total_expenses = float(today_df["expenses_amount"].sum())
total_waiting_pay = float(today_df["waiting_amount"].sum())
total_add_pay = float(today_df["add_pay"].sum())

gross_today = total_amount + total_waiting_pay + total_add_pay
net_today = gross_today - total_expenses
gap = TARGET - net_today

avg_per_car = gross_today / cars_driven if cars_driven > 0 else 0.0
waiting_rate = cfg.WAITING_RATE

cars_needed = math.ceil(gap / avg_per_car) if gap > 0 and avg_per_car > 0 else 0
waiting_hours_needed = gap / waiting_rate if gap > 0 and waiting_rate > 0 else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Cars Driven Today", cars_driven)
col2.metric("Net Today", f"£{net_today:.2f}")
col3.metric("Gap to £{:.0f}".format(TARGET), f"£{gap:.2f}" if gap > 0 else "£0.00")

col4, col5, col6 = st.columns(3)
col4.metric("Job Amount", f"£{total_amount:.2f}")
col5.metric("Waiting Pay", f"£{total_waiting_pay:.2f}")
col6.metric("Expenses", f"£{total_expenses:.2f}")

st.metric("Add Pay", f"£{total_add_pay:.2f}")
st.write(f"**Average earned per car:** £{avg_per_car:.2f}")

if gap > 0:
    st.warning(f"You are £{gap:.2f} short of your £{TARGET:.0f} target.")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("More Cars Needed", f"{cars_needed}" if cars_needed > 0 else "0")
    col_b.metric("More Waiting Time Needed", f"{waiting_hours_needed:.1f} hrs")
    col_c.metric("Expense Reduction Needed", f"£{gap:.2f}")

    st.info(
        f"To hit £{TARGET:.0f}, you could add about {cars_needed} more average car(s), "
        f"or about {waiting_hours_needed:.1f} more waiting hour(s), "
        f"or reduce expenses by £{gap:.2f}."
    )
else:
    st.success("Target hit. Nice.")

st.subheader("This Week Overview")

start_of_week = today - timedelta(days=today.weekday())
week_df = df[df["work_date"] >= start_of_week].copy()

for col in ["amount", "expenses_amount", "waiting_amount", "add_pay"]:
    if col in week_df.columns:
        week_df[col] = pd.to_numeric(week_df[col], errors="coerce").fillna(0.0)
    else:
        week_df[col] = 0.0

# Count only real completed driving jobs this week
if "job_outcome" in week_df.columns:
    weekly_cars_df = week_df[
        week_df["job_outcome"].fillna("").str.strip().str.lower() == "completed"
    ].copy()
else:
    weekly_cars_df = week_df.copy()

weekly_cars_df = weekly_cars_df[weekly_cars_df["amount"] > 0].copy()

if "category" in weekly_cars_df.columns:
    weekly_cars_df = weekly_cars_df[
        weekly_cars_df["category"].fillna("").str.strip().str.lower() == "strd trade plate"
    ].copy()

if "job_id" in weekly_cars_df.columns:
    weekly_cars = (
        weekly_cars_df["job_id"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )
else:
    weekly_cars = len(weekly_cars_df)

weekly_amount = float(week_df["amount"].sum())
weekly_waiting = float(week_df["waiting_amount"].sum())
weekly_add_pay = float(week_df["add_pay"].sum())
weekly_expenses = float(week_df["expenses_amount"].sum())
weekly_net = weekly_amount + weekly_waiting + weekly_add_pay - weekly_expenses

weekly_target = cfg.WEEKLY_TARGET
weekly_gap = weekly_target - weekly_net

debug_cols = [col for col in [
    "work_date",
    "job_id",
    "category",
    "amount",
    "job_outcome",
    "job_status",
    "description"
] if col in weekly_cars_df.columns]

st.subheader("Weekly Cars Debug")
st.dataframe(weekly_cars_df[debug_cols].sort_values(by="work_date", ascending=False))

col7, col8, col9, col10 = st.columns(4)
col7.metric("Cars Driven This Week", weekly_cars)
col8.metric("Weekly Net", f"£{weekly_net:.2f}")
col9.metric("Weekly Expenses", f"£{weekly_expenses:.2f}")
col10.metric("Gap to Weekly Target", f"£{weekly_gap:.2f}" if weekly_gap > 0 else "£0.00")