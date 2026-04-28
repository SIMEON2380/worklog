import os
import streamlit as st
import requests
from datetime import date

st.title("API Add Entry (Test)")

# API config
API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("WORKLOG_API_KEY", "")

# form
work_date = st.date_input("Work Date", value=date.today())
job_id = st.text_input("Job ID")

category = st.selectbox(
    "Job Type",
    ["STRD Trade Plate", "Inspect and Collect", "Inspect and Collect 2"]
)

vehicle_description = st.text_input("Vehicle Description")
vehicle_reg = st.text_input("Vehicle Reg")

collection_from = st.text_input("Collection From")
delivery_to = st.text_input("Delivery To")

amount = st.number_input("Amount", min_value=0.0)

job_status = st.selectbox(
    "Job Status",
    ["Start", "Pending", "Paid", "Completed", "Aborted", "Withdraw"]
)

waiting_time = st.text_input("Waiting Time (e.g. 10-11)")
add_pay = st.number_input("Add Pay", min_value=0.0)

job_expenses = st.selectbox(
    "Job Expenses",
    ["No expenses", "uber", "taxi", "train", "toll", "fuel", "other"]
)

expenses_amount = st.number_input("Expenses Amount", min_value=0.0)

auth_code = st.text_input("Auth Code")
comments = st.text_input("Comments")

# button
if st.button("Save via API"):

    data = {
        "work_date": str(work_date),
        "job_id": job_id,
        "category": category,
        "vehicle_description": vehicle_description,
        "vehicle_reg": vehicle_reg.upper() if vehicle_reg else None,
        "collection_from": collection_from,
        "delivery_to": delivery_to,
        "amount": amount,
        "job_status": job_status,
        "waiting_time": waiting_time,
        "add_pay": add_pay,
        "job_expenses": job_expenses,
        "expenses_amount": expenses_amount,
        "auth_code": auth_code,
        "comments": comments
    }

    try:
        response = requests.post(
            f"{API_URL}/jobs",
            json=data,
            headers={
                "x-api-key": API_KEY
            }
        )

        if response.status_code == 201:
            st.success("Job saved via API ✅")
            st.write(response.json())
        else:
            st.error(f"Failed: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error(f"Error: {e}")