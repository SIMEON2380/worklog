import streamlit as st
import requests
from datetime import date

st.title("API Add Entry (Test)")

# API config
API_URL = "http://127.0.0.1:8000"
API_KEY = "supersecret123"

# form
work_date = st.date_input("Work Date", value=date.today())
job_id = st.text_input("Job ID")
amount = st.number_input("Amount", min_value=0.0)
job_status = st.selectbox(
    "Job Status",
    ["Start", "Pending", "Paid", "Completed", "Aborted", "Withdraw"]
)

# button
if st.button("Save via API"):
    data = {
        "work_date": str(work_date),
        "job_id": job_id,
        "amount": amount,
        "job_status": job_status
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