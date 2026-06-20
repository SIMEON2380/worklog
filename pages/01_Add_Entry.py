import os
from datetime import date

import requests
import streamlit as st

from worklog.auth import ensure_default_user
from worklog.config import Config
from worklog.ui import require_login

API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("WORKLOG_API_KEY", "")

cfg = Config()

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Add Entry", layout="wide")

ensure_default_user(cfg)
require_login()

st.markdown(
    """
    <style>
        .section-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 14px;
            padding: 1.2rem;
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
            color: #ffffff;
        }

        .summary-card {
            background: linear-gradient(135deg, #161b22, #1f2937);
            border: 1px solid #374151;
            border-radius: 16px;
            padding: 1.2rem;
            margin-bottom: 1rem;
        }

        .summary-row {
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #30363d;
            padding: 0.45rem 0;
            font-size: 0.95rem;
        }

        .summary-total {
            display: flex;
            justify-content: space-between;
            padding-top: 0.8rem;
            font-size: 1.15rem;
            font-weight: 800;
            color: #ffffff;
        }

        .muted {
            color: #9ca3af;
        }

        div.stButton > button:first-child {
            border-radius: 10px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Add New Job")


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


if "expense_rows" not in st.session_state:
    st.session_state.expense_rows = [{"type": "No expenses", "amount": 0.0}]


left_col, right_col = st.columns([2.2, 1])


with left_col:
    with st.container(border=True):
        st.markdown("### 📋 Job Information")

        col1, col2, col3 = st.columns(3)
        work_date = col1.date_input("Date", value=date.today())
        job_number = col2.text_input("Job Number")
        job_type = col3.selectbox("Job Type", cfg.JOB_TYPE_OPTIONS)

        col4, col5 = st.columns(2)
        job_outcome = col4.selectbox(
            "Job Outcome",
            ["Completed", "Aborted", "Withdraw", "Fail"],
            index=0,
        )
        job_status = col5.selectbox("Job Status", cfg.STATUS_OPTIONS)

    with st.container(border=True):
        st.markdown("### 🚗 Vehicle Details")

        col1, col2 = st.columns(2)
        vehicle_description = col1.text_input("Vehicle Description")
        vehicle_reg = col2.text_input("Vehicle Reg")

    with st.container(border=True):
        st.markdown("### 📍 Journey")

        col1, col2 = st.columns(2)
        collection_from = col1.text_input("Collection From")
        delivery_to = col2.text_input("Delivery To")

    with st.container(border=True):
        st.markdown("### 💰 Job Pay & Expenses")

        job_amount = st.number_input("Job Amount (£)", min_value=0.0, step=1.0)

        st.markdown("#### Expenses")

        for i, expense in enumerate(st.session_state.expense_rows):
            exp_col1, exp_col2, exp_col3 = st.columns([2, 1, 0.7])

            expense_type = exp_col1.selectbox(
                f"Expense Type {i + 1}",
                cfg.JOB_EXPENSE_OPTIONS,
                index=cfg.JOB_EXPENSE_OPTIONS.index(expense["type"])
                if expense["type"] in cfg.JOB_EXPENSE_OPTIONS
                else 0,
                key=f"expense_type_{i}",
            )

            expense_amount = exp_col2.number_input(
                f"Amount {i + 1} (£)",
                min_value=0.0,
                step=0.5,
                value=float(expense.get("amount", 0.0)),
                key=f"expense_amount_{i}",
            )

            st.session_state.expense_rows[i] = {
                "type": expense_type,
                "amount": float(expense_amount),
            }

            with exp_col3:
                st.write("")
                st.write("")
                if len(st.session_state.expense_rows) > 1:
                    if st.button("Remove", key=f"remove_expense_{i}"):
                        st.session_state.expense_rows.pop(i)
                        st.rerun()

        if st.button("➕ Add Another Expense"):
            st.session_state.expense_rows.append(
                {"type": "No expenses", "amount": 0.0}
            )
            st.rerun()

    valid_expenses = [
        row
        for row in st.session_state.expense_rows
        if row["type"] != "No expenses" and float(row["amount"]) > 0
    ]

    expenses_amount = round(
        sum(float(row["amount"]) for row in valid_expenses),
        2,
    )

    job_expenses = (
        "; ".join(
            f"{row['type']}: £{float(row['amount']):.2f}"
            for row in valid_expenses
        )
        if valid_expenses
        else "No expenses"
    )

    with st.container(border=True):
        st.markdown("### ⏱ Waiting Time")

        col1, col2, col3 = st.columns(3)

        waiting_time = col1.text_input(
            "Waiting Time",
            placeholder="e.g. 10-11 or 09:00-11:30",
        )

        calc_waiting_hours = float(parse_wait_range_to_hours(waiting_time))
        calc_waiting_amount = round(
            float(calc_waiting_hours) * float(getattr(cfg, "WAITING_RATE", 0.0)),
            2,
        )

        col2.metric("Waiting Hours", f"{calc_waiting_hours:.2f}")
        col3.metric("Waiting Amount", f"£{calc_waiting_amount:.2f}")

    with st.container(border=True):
        st.markdown("### 📝 Extra Details")

        col1, col2, col3 = st.columns(3)
        add_pay = col1.number_input("Add Pay (£)", min_value=0.0, step=1.0, value=0.0)
        auth_code = col2.text_input("Auth Code")
        comments = col3.text_input("Comments")


total_to_pay = round(
    float(job_amount)
    + float(calc_waiting_amount)
    + float(add_pay)
    - float(expenses_amount),
    2,
)


with right_col:
    with st.container(border=True):
        st.markdown("### 📊 Financial Summary")

        st.metric("Total To Pay", f"£{total_to_pay:.2f}")

        st.divider()

        st.write(f"**Job Amount:** £{float(job_amount):.2f}")
        st.write(f"**Waiting Amount:** £{float(calc_waiting_amount):.2f}")
        st.write(f"**Add Pay:** £{float(add_pay):.2f}")
        st.write(f"**Expenses:** £{float(expenses_amount):.2f}")

        st.divider()

        if valid_expenses:
            st.markdown("#### Expense Breakdown")
            for row in valid_expenses:
                st.write(f"- {row['type']}: £{float(row['amount']):.2f}")
        else:
            st.caption("No expenses added.")

    save_clicked = st.button("💾 Save Job", type="primary", use_container_width=True)


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
            "expenses_amount": float(expenses_amount),
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

            if response.status_code in (200, 201):
                st.success(f"Job {clean_job_number} saved via API ✅")
                st.session_state.expense_rows = [{"type": "No expenses", "amount": 0.0}]
                st.rerun()
            elif response.status_code == 400:
                st.error("Validation failed.")
                st.write(api_error_message(response))
            elif response.status_code == 401:
                st.error("API key rejected. Check WORKLOG_API_KEY.")
            elif response.status_code == 404:
                st.error("API endpoint not found.")
            elif response.status_code == 409:
                st.error(f"Job {clean_job_number} already exists.")
            elif response.status_code == 422:
                st.error("API validation error.")
                st.write(api_error_message(response))
            else:
                st.error(f"API failed: {response.status_code}")
                st.write(api_error_message(response))

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to API at {API_URL}")
        except requests.exceptions.Timeout:
            st.error("API request timed out.")
        except Exception as e:
            st.error(f"Save failed: {e}")