import os
from datetime import date

import requests
import streamlit as st

from worklog.auth import ensure_default_user
from worklog.config import Config
from worklog.db import make_db
from worklog.ui import require_login, check_vehicle_compliance

API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("WORKLOG_API_KEY", "supersecret123")

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Add Entry", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Add New Job")


def parse_wait_range_to_hours(s: str) -> float:
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
        return round((b - a) / 60.0, 2)
    except Exception:
        return 0.0


def clean_text(value):
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def api_error_message(response):
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get("detail") or payload.get("message") or str(payload)
        return str(payload)
    except Exception:
        return response.text


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

vehicle_check = check_vehicle_compliance(vehicle_reg)

if vehicle_reg and str(vehicle_reg).strip():
    mot_expiry = vehicle_check.get("mot_expiry")
    mot_expiry_text = mot_expiry.isoformat() if mot_expiry else "Unknown"

    col5.caption(
        f"Road Tax: {vehicle_check.get('tax_status', 'Unknown')} | "
        f"MOT Expiry: {mot_expiry_text}"
    )

    status = vehicle_check.get("status", "Unknown")
    reason = vehicle_check.get("reason", "")

    if status == "Compliant":
        col5.success("Compliant")
    elif status == "Warning":
        col5.warning(f"Warning - {reason}")
    elif status == "Non-compliant":
        col5.error(f"Non-compliant - {reason}")
    else:
        col5.info(f"{status}" + (f" - {reason}" if reason else ""))

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

calc_waiting_hours = float(parse_wait_range_to_hours(waiting_time))
calc_waiting_amount = round(float(calc_waiting_hours) * float(getattr(cfg, "WAITING_RATE", 0.0)), 2)

col14.number_input(
    "Waiting Hours (auto)",
    min_value=0.0,
    step=0.5,
    value=float(calc_waiting_hours),
    disabled=True,
)

col15.number_input(
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

save_clicked = st.button("Save Job", type="primary")

if save_clicked:
    clean_work_date = work_date.isoformat() if work_date else None
    clean_job_number = clean_text(job_number)
    clean_vehicle_description = clean_text(vehicle_description)
    clean_vehicle_reg = clean_text(vehicle_reg)
    clean_collection_from = clean_text(collection_from)
    clean_delivery_to = clean_text(delivery_to)
    clean_auth_code = clean_text(auth_code)
    clean_waiting_time = clean_text(waiting_time)
    clean_comments = clean_text(comments)

    if clean_vehicle_reg:
        clean_vehicle_reg = clean_vehicle_reg.upper()

    missing_fields = []

    if not clean_work_date:
        missing_fields.append("Date")
    if not clean_job_number:
        missing_fields.append("Job Number")
    if not clean_vehicle_description:
        missing_fields.append("Vehicle Description")
    if float(job_amount) <= 0:
        missing_fields.append("Job Amount")

    if missing_fields:
        st.error("Please complete these required fields: " + ", ".join(missing_fields))
    else:
        payload = {
            "work_date": clean_work_date,
            "job_id": clean_job_number,
            "amount": float(job_amount),
            "category": job_type,
            "job_status": job_status,
            "waiting_time": clean_waiting_time,
            "waiting_hours": float(calc_waiting_hours),
            "waiting_amount": float(calc_waiting_amount),
            "vehicle_description": clean_vehicle_description,
            "vehicle_reg": clean_vehicle_reg,
            "collection_from": clean_collection_from,
            "delivery_to": clean_delivery_to,
            "job_expenses": job_expenses,
            "expenses_amount": float(expenses_amount) if expenses_amount is not None else 0.0,
            "auth_code": clean_auth_code,
            "comments": clean_comments,
            "add_pay": float(add_pay),
            "job_outcome": job_outcome,
        }

        try:
            response = requests.post(
                f"{API_URL}/jobs",
                json=payload,
                headers={"x-api-key": API_KEY},
                timeout=15,
            )

            if response.status_code == 201:
                st.success(f"Job {clean_job_number} saved via API ✅")
                st.rerun()
            elif response.status_code == 409:
                st.error(f"Job {clean_job_number} already exists.")
            elif response.status_code == 401:
                st.error("API key rejected. Check WORKLOG_API_KEY.")
            else:
                st.error(f"API failed: {response.status_code}")
                st.write(api_error_message(response))

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to API at {API_URL}")
        except requests.exceptions.Timeout:
            st.error("API request timed out.")
        except Exception as e:
            st.error(f"Save failed: {e}")