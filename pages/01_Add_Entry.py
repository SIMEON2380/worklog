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
    waiting_time = col12.text_input("Waiting Time (e.g. 09:00-11:30)")

    auth_code = st.text_input("Auth Code")
    comments = st.text_area("Comments")

    submitted = st.form_submit_button("Save Job")

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
                "amount": float(job_amount) if job_amount is not None else None,
                "job_expenses": job_expenses,
                "expenses_amount": float(expenses_amount) if expenses_amount is not None else None,
                "auth_code": auth_code.strip() if auth_code else None,
                "job_status": job_status,
                "status": job_status,  # keep both columns aligned (your schema has both)
                "waiting_time": waiting_time.strip() if waiting_time else None,
                "comments": comments.strip() if comments else None,
            }
        )

        st.success("Job saved successfully.")
        st.rerun()