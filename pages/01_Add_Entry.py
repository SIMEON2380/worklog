import re
import streamlit as st
from datetime import date
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Add Entry", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Add New Job")


# -------------------------
# Helpers
# -------------------------
def parse_wait_range_to_hours(s: str) -> float:
    """
    Accepts:
      "10-11" -> 1.0
      "10:30-12:00" -> 1.5
      "10 - 11" -> 1.0
    Returns 0.0 if invalid.
    """
    if not s:
        return 0.0

    s = str(s).strip().replace(" ", "")
    if "-" not in s:
        return 0.0

    start, end = s.split("-", 1)

    def to_minutes(t: str) -> int:
        if ":" in t:
            hh, mm = t.split(":", 1)
            return int(hh) * 60 + int(mm)
        return int(t) * 60

    try:
        a = to_minutes(start)
        b = to_minutes(end)
        if b <= a:
            return 0.0
        return (b - a) / 60.0
    except Exception:
        return 0.0


def normalise_postcode(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    return text


def find_postcode_history(df: pd.DataFrame, postcode: str) -> pd.DataFrame:
    if df.empty or not postcode:
        return pd.DataFrame()

    if "postcode" not in df.columns:
        return pd.DataFrame()

    sub = df.copy()

    sub["postcode_norm"] = (
        sub["postcode"]
        .fillna("")
        .astype(str)
        .apply(normalise_postcode)
    )

    target = normalise_postcode(postcode)
    matches = sub[sub["postcode_norm"] == target].copy()

    if matches.empty:
        return matches

    if "work_date" in matches.columns:
        matches["work_date"] = pd.to_datetime(matches["work_date"], errors="coerce")
        matches = matches.sort_values("work_date", ascending=False)

    return matches


def show_postcode_history_box(df_all: pd.DataFrame, entered_postcode: str) -> None:
    entered_postcode = str(entered_postcode or "").strip()

    if not entered_postcode:
        return

    matches = find_postcode_history(df_all, entered_postcode)

    if matches.empty:
        st.info("No previous jobs found for this postcode.")
        return

    times_visited = len(matches)

    last_visited = "N/A"
    if "work_date" in matches.columns and matches["work_date"].notna().any():
        last_dt = matches["work_date"].dropna().iloc[0]
        last_visited = last_dt.strftime("%Y-%m-%d")

    last_vehicle = "N/A"
    if "vehicle_description" in matches.columns:
        vals = matches["vehicle_description"].fillna("").astype(str).str.strip()
        vals = vals[vals != ""]
        if not vals.empty:
            last_vehicle = vals.iloc[0]

    last_job_type = "N/A"
    if "category" in matches.columns:
        vals = matches["category"].fillna("").astype(str).str.strip()
        vals = vals[vals != ""]
        if not vals.empty:
            last_job_type = vals.iloc[0]

    last_comment = "N/A"
    comment_col = None
    for col in ["comments", "description", "comment", "notes"]:
        if col in matches.columns:
            comment_col = col
            break

    if comment_col:
        vals = matches[comment_col].fillna("").astype(str).str.strip()
        vals = vals[vals != ""]
        if not vals.empty:
            last_comment = vals.iloc[0]

    st.warning(f"You've been here before: {times_visited} time(s).")

    c1, c2, c3 = st.columns(3)
    c1.metric("Times Visited", times_visited)
    c2.metric("Last Visited", last_visited)
    c3.metric("Last Job Type", last_job_type)

    c4, c5 = st.columns(2)
    c4.metric("Last Vehicle", last_vehicle)
    c5.metric("Postcode", entered_postcode.upper())

    if last_comment != "N/A":
        st.caption(f"Last note: {last_comment}")

    with st.expander("Show previous jobs for this postcode"):
        show_cols = [
            "work_date",
            "job_id",
            "category",
            "vehicle_description",
            "collection_from",
            "delivery_to",
            "job_status",
            "postcode",
        ]
        available_cols = [c for c in show_cols if c in matches.columns]
        st.dataframe(matches[available_cols], use_container_width=True, hide_index=True)


# Load all rows once for postcode lookup
df_all = DB["read_all"]().copy()

with st.form("add_job_form"):
    col1, col2, col3 = st.columns(3)

    work_date = col1.date_input("Date", value=date.today())
    job_number = col2.text_input("Job Number")
    job_type = col3.selectbox("Job Type", cfg.JOB_TYPE_OPTIONS)

    col4, col5, col6 = st.columns(3)
    vehicle_description = col4.text_input("Vehicle Description")
    vehicle_reg = col5.text_input("Vehicle Reg")
    job_status = col6.selectbox("Job Status", cfg.STATUS_OPTIONS)

    col7, col8, col9 = st.columns(3)
    collection_from = col7.text_input("Collection From")
    delivery_to = col8.text_input("Delivery To")
    job_amount = col9.number_input("Job Amount (£)", min_value=0.0, step=1.0)

    col10, col11, col12 = st.columns(3)
    job_expenses = col10.selectbox("Job Expenses", cfg.JOB_EXPENSE_OPTIONS)
    expenses_amount = col11.number_input("Expenses Amount (£)", min_value=0.0, step=0.5)
    waiting_time = col12.text_input("Waiting Time (e.g. 10-11 or 09:00-11:30)")

    # NEW: Postcode field
    postcode = st.text_input("Postcode")

    # Auto-calc waiting from waiting_time (keeps reports consistent)
    calc_waiting_hours = float(parse_wait_range_to_hours(waiting_time))
    calc_waiting_amount = float(calc_waiting_hours) * float(getattr(cfg, "WAITING_RATE", 0.0))

    col13, col14, col15 = st.columns(3)
    waiting_hours = col13.number_input(
        "Waiting Hours (auto)",
        min_value=0.0,
        step=0.5,
        value=float(calc_waiting_hours),
        disabled=True,
    )
    waiting_amount = col14.number_input(
        "Waiting Amount (£) (auto)",
        min_value=0.0,
        step=0.5,
        value=float(calc_waiting_amount),
        disabled=True,
    )

    add_pay = col15.number_input("Add Pay (£)", min_value=0.0, step=1.0, value=0.0)

    auth_code = st.text_input("Auth Code")
    comments = st.text_area("Comments")

    submitted = st.form_submit_button("Save Job", type="primary")

    if submitted:
        DB["insert_row"](
            {
                "work_date": work_date.isoformat() if work_date else None,
                "job_id": job_number.strip() if job_number else None,
                "category": job_type,
                "vehicle_description": vehicle_description.strip() if vehicle_description else None,
                "vehicle_reg": vehicle_reg.strip() if vehicle_reg else None,
                "collection_from": collection_from.strip() if collection_from else None,
                "delivery_to": delivery_to.strip() if delivery_to else None,
                "postcode": postcode.strip().upper() if postcode else None,
                "amount": float(job_amount) if job_amount is not None else 0.0,
                "job_expenses": job_expenses,
                "expenses_amount": float(expenses_amount) if expenses_amount is not None else 0.0,
                "auth_code": auth_code.strip() if auth_code else None,
                "job_status": job_status,
                "status": job_status,
                "waiting_time": waiting_time.strip() if waiting_time else None,
                "waiting_hours": float(calc_waiting_hours),
                "waiting_amount": float(calc_waiting_amount),
                "add_pay": float(add_pay),
                "comments": comments.strip() if comments else None,
            }
        )

        st.success("Job saved successfully.")
        st.rerun()


# Show postcode history outside the form so it updates before save
show_postcode_history_box(df_all, locals().get("postcode", ""))