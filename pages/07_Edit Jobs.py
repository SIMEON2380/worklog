import os
from datetime import date

import pandas as pd
import requests
import streamlit as st

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login

API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("WORKLOG_API_KEY", "supersecret123")

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Edit Jobs", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Edit Jobs (Form)")


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


if "edit_job_search" not in st.session_state:
    st.session_state.edit_job_search = ""

if "edit_job_choice" not in st.session_state:
    st.session_state.edit_job_choice = None

if "clear_edit_form_after_save" not in st.session_state:
    st.session_state.clear_edit_form_after_save = False

if st.session_state.clear_edit_form_after_save:
    st.session_state.edit_job_search = ""
    st.session_state.edit_job_choice = None
    st.session_state.clear_edit_form_after_save = False


def load_jobs_df() -> pd.DataFrame:
    res = requests.get(
        f"{API_URL}/jobs",
        headers={"x-api-key": API_KEY},
        params={"page": 1, "page_size": 500},
        timeout=20,
    )
    res.raise_for_status()
    payload = res.json()

    if isinstance(payload, dict) and "data" in payload:
        return pd.DataFrame(payload.get("data", []))

    if isinstance(payload, list):
        return pd.DataFrame(payload)

    return pd.DataFrame()


try:
    df = load_jobs_df().copy()
except Exception as e:
    st.error(f"Failed to load jobs from API: {e}")
    st.stop()

if df.empty:
    st.info("No jobs found.")
    st.stop()

if "id" not in df.columns:
    st.error("Missing 'id' column in dataset. Cannot edit safely.")
    st.stop()

if "work_date" in df.columns:
    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date

if "paid_date" in df.columns:
    df["paid_date"] = pd.to_datetime(df["paid_date"], errors="coerce").dt.date

STATUS_COL = "job_status" if "job_status" in df.columns else ("status" if "status" in df.columns else None)
OUTCOME_OPTIONS = ["Completed", "Aborted", "Withdraw", "Fail"]

left, right = st.columns([2, 3])

job_search = left.text_input(
    "Search by Job Number / Vehicle Reg (optional)",
    key="edit_job_search",
)

filtered = df.copy()

if job_search.strip():
    s = job_search.strip().lower()
    mask = pd.Series(False, index=filtered.index)

    if "job_id" in filtered.columns:
        mask = mask | filtered["job_id"].astype(str).str.lower().str.contains(s, na=False)

    if "vehicle_reg" in filtered.columns:
        mask = mask | filtered["vehicle_reg"].astype(str).str.lower().str.contains(s, na=False)

    filtered = filtered[mask]

if filtered.empty:
    st.warning("No matching jobs.")
    st.stop()


def label_row(r) -> str:
    jid = str(r.get("job_id", "") or "")
    vreg = str(r.get("vehicle_reg", "") or "")
    wdt = r.get("work_date", None)
    d = wdt.isoformat() if hasattr(wdt, "isoformat") else str(wdt or "")
    return f"#{int(r['id'])} | {jid} | {vreg} | {d}"


rows = filtered.sort_values(by="id", ascending=False).to_dict("records")
labels = [label_row(r) for r in rows]

options = [None] + list(range(len(rows)))

if st.session_state.edit_job_choice not in options:
    st.session_state.edit_job_choice = None

choice = right.selectbox(
    "Select job to edit",
    options=options,
    index=options.index(st.session_state.edit_job_choice),
    format_func=lambda i: "Select a job..." if i is None else labels[i],
    key="edit_job_choice",
)

if choice is None:
    st.info("Select a job to edit.")
    st.stop()

job = rows[choice]
row_id = int(job["id"])

st.caption(f"Selected Row ID: {row_id}")
st.divider()

with st.form("edit_job_form"):
    col1, col2, col3 = st.columns(3)

    work_date = col1.date_input("Date", value=job.get("work_date") or date.today())
    job_number = col2.text_input("Job Number", value=str(job.get("job_id") or ""))
    job_type = col3.selectbox(
        "Job Type",
        cfg.JOB_TYPE_OPTIONS,
        index=(cfg.JOB_TYPE_OPTIONS.index(job.get("category")) if job.get("category") in cfg.JOB_TYPE_OPTIONS else 0),
    )

    col4, col5, col6 = st.columns(3)
    vehicle_description = col4.text_input("Vehicle Description", value=str(job.get("vehicle_description") or ""))
    vehicle_reg = col5.text_input("Vehicle Reg", value=str(job.get("vehicle_reg") or ""))

    current_outcome = str(job.get("job_outcome") or "Completed")
    if current_outcome not in OUTCOME_OPTIONS:
        current_outcome = "Completed"

    job_outcome = col6.selectbox(
        "Job Outcome",
        OUTCOME_OPTIONS,
        index=OUTCOME_OPTIONS.index(current_outcome),
    )

    col7, col8, col9 = st.columns(3)
    collection_from = col7.text_input("Collection From", value=str(job.get("collection_from") or ""))
    delivery_to = col8.text_input("Delivery To", value=str(job.get("delivery_to") or ""))

    current_status = str(job.get(STATUS_COL) or "Pending") if STATUS_COL else "Pending"
    job_status = col9.selectbox(
        "Job Status",
        cfg.STATUS_OPTIONS,
        index=(cfg.STATUS_OPTIONS.index(current_status) if current_status in cfg.STATUS_OPTIONS else 0),
    )

    col10, col11, col12 = st.columns(3)
    job_amount = col10.number_input(
        "Job Amount (£)",
        min_value=0.0,
        step=1.0,
        value=float(job.get("amount") or 0.0),
    )
    job_expenses = col11.selectbox(
        "Job Expenses",
        cfg.JOB_EXPENSE_OPTIONS,
        index=(cfg.JOB_EXPENSE_OPTIONS.index(job.get("job_expenses")) if job.get("job_expenses") in cfg.JOB_EXPENSE_OPTIONS else 0),
    )
    expenses_amount = col12.number_input(
        "Expenses Amount (£)",
        min_value=0.0,
        step=0.5,
        value=float(job.get("expenses_amount") or 0.0),
    )

    col13, col14, col15 = st.columns(3)
    waiting_time = col13.text_input(
        "Waiting Time (e.g. 10-11 or 09:00-11:30)",
        value=str(job.get("waiting_time") or ""),
    )

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

    add_pay_value = float(job.get("add_pay") or 0.0) if "add_pay" in df.columns else 0.0
    add_pay = col16.number_input(
        "Add Pay (£)",
        min_value=0.0,
        step=1.0,
        value=add_pay_value,
    )

    hours = col17.number_input(
        "Hours (if used)",
        min_value=0.0,
        step=0.5,
        value=float(job.get("hours") or 0.0) if "hours" in df.columns else 0.0,
        disabled=("hours" not in df.columns),
        help=None if "hours" in df.columns else "DB/API does not currently expose an hours field.",
    )

    existing_paid_date = job.get("paid_date") if "paid_date" in df.columns else None
    if pd.isna(existing_paid_date):
        existing_paid_date = None

    default_paid_date = existing_paid_date or date.today()

    paid_date = col18.date_input(
        "Paid Date",
        value=default_paid_date,
        disabled=(("paid_date" not in df.columns) or (str(job_status).strip().lower() != "paid")),
        help=(
            "Auto-fills to today when status is Paid, but you can change it manually."
            if "paid_date" in df.columns
            else "DB/API does not currently expose a paid_date field."
        ),
    )

    auth_code = st.text_input("Auth Code", value=str(job.get("auth_code") or ""))
    comments = st.text_area("Comments", value=str(job.get("comments") or ""))

    submitted = st.form_submit_button("Save changes", type="primary")

    if submitted:
        diffs = {}

        def set_if_changed(db_col: str, new_val):
            old_val = job.get(db_col)

            if isinstance(old_val, str):
                old_cmp = old_val.strip()
            else:
                old_cmp = old_val

            if isinstance(new_val, str):
                new_cmp = new_val.strip()
            else:
                new_cmp = new_val

            if pd.isna(old_cmp) and pd.isna(new_cmp):
                return
            if old_cmp == new_cmp:
                return

            diffs[db_col] = new_val

        set_if_changed("work_date", work_date.isoformat() if work_date else None)
        set_if_changed("job_id", clean_text(job_number))
        set_if_changed("category", job_type)
        set_if_changed("vehicle_description", clean_text(vehicle_description))
        set_if_changed("vehicle_reg", clean_text(vehicle_reg).upper() if clean_text(vehicle_reg) else None)
        set_if_changed("job_outcome", job_outcome)
        set_if_changed("collection_from", clean_text(collection_from))
        set_if_changed("delivery_to", clean_text(delivery_to))

        set_if_changed("amount", float(job_amount))
        set_if_changed("job_expenses", job_expenses)
        set_if_changed("expenses_amount", float(expenses_amount))

        set_if_changed("waiting_time", clean_text(waiting_time))
        set_if_changed("waiting_hours", float(calc_waiting_hours))
        set_if_changed("waiting_amount", float(calc_waiting_amount))
        set_if_changed("add_pay", float(add_pay))

        if "hours" in df.columns:
            set_if_changed("hours", float(hours))

        if "paid_date" in df.columns:
            if str(job_status).strip().lower() == "paid":
                set_if_changed("paid_date", paid_date.isoformat() if paid_date else date.today().isoformat())
            else:
                set_if_changed("paid_date", None)

        set_if_changed("auth_code", clean_text(auth_code))
        set_if_changed("comments", clean_text(comments))

        if STATUS_COL:
            set_if_changed("job_status", job_status)

        if diffs:
            try:
                response = requests.put(
                    f"{API_URL}/jobs/row/{row_id}",
                    json=diffs,
                    headers={"x-api-key": API_KEY},
                    timeout=20,
                )

                if response.status_code == 200:
                    st.session_state.clear_edit_form_after_save = True
                    st.success("Saved via API. Reports will reflect this immediately.")
                    st.rerun()
                else:
                    st.error(f"API update failed: {response.status_code}")
                    st.write(api_error_message(response))

            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to API at {API_URL}")
            except requests.exceptions.Timeout:
                st.error("API request timed out.")
            except Exception as e:
                st.error(f"Save failed: {e}")
        else:
            st.info("No changes detected.")

st.divider()

st.markdown("### Delete Selected Job")
st.error("This permanently deletes the selected job from the database.")

delete_confirm = st.checkbox(f"I confirm I want to delete job row #{row_id}")

if st.button("Delete selected job"):
    if not delete_confirm:
        st.warning("Tick the confirmation box before deleting.")
    else:
        try:
            response = requests.delete(
                f"{API_URL}/jobs/row/{row_id}",
                headers={"x-api-key": API_KEY},
                timeout=20,
            )

            if response.status_code == 200:
                st.session_state.clear_edit_form_after_save = True
                st.success(f"Job row #{row_id} deleted successfully via API.")
                st.rerun()
            else:
                st.error(f"API delete failed: {response.status_code}")
                st.write(api_error_message(response))

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to API at {API_URL}")
        except requests.exceptions.Timeout:
            st.error("API request timed out.")
        except Exception as e:
            st.error(f"Delete failed: {e}")