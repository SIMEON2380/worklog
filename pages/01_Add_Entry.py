import streamlit as st
from datetime import date

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


with st.form("add_job_form"):
    col1, col2, col3 = st.columns(3)

    work_date = col1.date_input("Date", value=date.today())
    job_number = col2.text_input("Job Number")
    job_type = col3.selectbox("Job Type", cfg.JOB_TYPE_OPTIONS)

    col4, col5, col6 = st.columns(3)
    vehicle_description = col4.text_input("Vehicle Description")
    vehicle_reg = col5.text_input("Vehicle Reg")
    job_outcome = col6.selectbox(
        "Job Outcome",
        ["Completed", "Aborted", "Withdraw", "Fail"],
        index=0,
    )

    col7, col8, col9 = st.columns(3)
    collection_from = col7.text_input("Collection From")
    delivery_to = col8.text_input("Delivery To")
    job_status = col9.selectbox("Job Status", cfg.STATUS_OPTIONS)

    col10, col11, col12 = st.columns(3)
    job_amount = col10.number_input("Job Amount (£)", min_value=0.0, step=1.0)
    job_expenses = col11.selectbox("Job Expenses", cfg.JOB_EXPENSE_OPTIONS)
    expenses_amount = col12.number_input("Expenses Amount (£)", min_value=0.0, step=0.5)

    col13, col14, col15 = st.columns(3)
    waiting_time = col13.text_input("Waiting Time (e.g. 10-11 or 09:00-11:30)")

    # Auto-calc waiting from waiting_time (keeps reports consistent)
    calc_waiting_hours = float(parse_wait_range_to_hours(waiting_time))
    calc_waiting_amount = float(calc_waiting_hours) * float(getattr(cfg, "WAITING_RATE", 0.0))

    waiting_hours = col14.number_input(
        "Waiting Hours (auto)",
        min_value=0.0,
        step=0.5,
        value=float(calc_waiting_hours),
        disabled=True,
    )
    waiting_amount = col15.number_input(
        "Waiting Amount (£) (auto)",
        min_value=0.0,
        step=0.5,
        value=float(calc_waiting_amount),
        disabled=True,
    )

    col16, col17, col18 = st.columns(3)
    add_pay = col16.number_input("Add Pay (£)", min_value=0.0, step=1.0, value=0.0)
    auth_code = col17.text_input("Auth Code")
    comments = col18.text_input("Comments")

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
                "amount": float(job_amount) if job_amount is not None else 0.0,
                "job_expenses": job_expenses,
                "expenses_amount": float(expenses_amount) if expenses_amount is not None else 0.0,
                "auth_code": auth_code.strip() if auth_code else None,
                "job_outcome": job_outcome,
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