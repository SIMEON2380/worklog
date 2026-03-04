import re
import streamlit as st
import pandas as pd
from datetime import date

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login, display_jobs_table

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

# -------------------------
# Add-Pay (from comment column)
# -------------------------
ADD_PAY_RE = re.compile(
    r"(?:add[\s_-]*pay|addpay)\s*[:=]?\s*£?\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE
)

def extract_add_pay(value) -> float:
    """
    Extract add-pay amounts from comment text.
    Examples matched:
      "add-pay 10"
      "add pay: £12.50"
      "ADD_PAY=5"
    If multiple appear, sums them.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    text = str(value)
    matches = ADD_PAY_RE.findall(text)
    if not matches:
        return 0.0
    total = 0.0
    for m in matches:
        try:
            total += float(m)
        except ValueError:
            pass
    return float(total)

# Find a comment column safely (won't break if it doesn't exist)
comment_col = None
for c in ["comment", "comments", "Comment", "Comments", "note", "notes", "Note", "Notes"]:
    if c in sub.columns:
        comment_col = c
        break

if comment_col:
    add_pay_series = sub[comment_col].apply(extract_add_pay)
else:
    add_pay_series = pd.Series(0.0, index=sub.index)

total_add_pay = float(pd.to_numeric(add_pay_series, errors="coerce").fillna(0).sum())

# -------------------------
# Totals
# -------------------------
total_job_amount = float(pd.to_numeric(sub["amount"], errors="coerce").fillna(0).sum())
total_wait_hours = float(pd.to_numeric(sub["waiting_hours"], errors="coerce").fillna(0).sum())
total_wait_amount = float(pd.to_numeric(sub["waiting_amount"], errors="coerce").fillna(0).sum())
total_expenses = float(pd.to_numeric(sub["expenses_amount"], errors="coerce").fillna(0).sum())

# Keep your existing logic (still subtract expenses), just add Add-Pay
grand_total = total_job_amount + total_wait_amount + total_add_pay - total_expenses

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Job Amount", f"£{total_job_amount:,.2f}")
c2.metric("Waiting Hours", f"{total_wait_hours:,.2f} hrs")
c3.metric("Waiting Pay", f"£{total_wait_amount:,.2f}")
c4.metric("Add-Pay", f"£{total_add_pay:,.2f}")
c5.metric("Expenses", f"£{total_expenses:,.2f}")
c6.metric("Grand Total", f"£{grand_total:,.2f}")

st.divider()

display_jobs_table(cfg, sub, caption="Jobs in selected day")