import os
from datetime import date

import pandas as pd
import requests
import streamlit as st

from worklog.config import Config
from worklog.auth import ensure_default_user
from worklog.ui import require_login

API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("WORKLOG_API_KEY", "")

cfg = Config()

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Edit Jobs", layout="wide")

ensure_default_user(cfg)
require_login()

st.title("Edit Jobs")


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


def parse_expenses_for_edit(expense_text, fallback_amount=0.0):
    if not expense_text or str(expense_text).strip().lower() == "no expenses":
        return [{"type": "No expenses", "amount": 0.0}]

    rows = []

    for part in str(expense_text).split(";"):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            expense_type, amount_text = part.split(":", 1)
            expense_type = expense_type.strip()
            amount_text = amount_text.replace("£", "").replace(",", "").strip()

            try:
                amount = float(amount_text)
            except Exception:
                amount = 0.0

            rows.append({"type": expense_type, "amount": amount})

    if not rows and float(fallback_amount or 0.0) > 0:
        rows.append({"type": "Other", "amount": float(fallback_amount)})

    return rows or [{"type": "No expenses", "amount": 0.0}]


def build_expenses_from_rows(rows):
    valid_rows = [
        row
        for row in rows
        if row.get("type") != "No expenses" and float(row.get("amount") or 0.0) > 0
    ]

    total = round(sum(float(row["amount"]) for row in valid_rows), 2)

    expense_text = (
        "; ".join(
            f"{row['type']}: £{float(row['amount']):.2f}"
            for row in valid_rows
        )
        if valid_rows
        else "No expenses"
    )

    return expense_text, total


if "edit_job_search" not in st.session_state:
    st.session_state.edit_job_search = ""

if "edit_job_choice" not in st.session_state:
    st.session_state.edit_job_choice = None

if "clear_edit_form_after_save" not in st.session_state:
    st.session_state.clear_edit_form_after_save = False

if "edit_expense_rows" not in st.session_state:
    st.session_state.edit_expense_rows = [{"type": "No expenses", "amount": 0.0}]

if "edit_expense_row_id" not in st.session_state:
    st.session_state.edit_expense_row_id = None

if st.session_state.clear_edit_form_after_save:
    st.session_state.edit_job_search = ""
    st.session_state.edit_job_choice = None
    st.session_state.edit_expense_rows = [{"type": "No expenses", "amount": 0.0}]
    st.session_state.edit_expense_row_id = None
    st.session_state.clear_edit_form_after_save = False


def load_jobs_df(search: str = "") -> pd.DataFrame:
    params = {
        "page": 1,
        "page_size": 200,
    }

    if search.strip():
        params["search"] = search.strip()

    res = requests.get(
        f"{API_URL}/jobs",
        headers={"x-api-key": API_KEY},
        params=params,
        timeout=20,
    )

    res.raise_for_status()
    payload = res.json()

    if isinstance(payload, dict) and "data" in payload:
        return pd.DataFrame(payload.get("data", []))

    if isinstance(payload, dict) and "items" in payload:
        return pd.DataFrame(payload.get("items", []))

    if isinstance(payload, list):
        return pd.DataFrame(payload)

    return pd.DataFrame()


search_col, select_col = st.columns([2, 3])

job_search = search_col.text_input(
    "Search by Job Number / Vehicle Reg / Location / Comments / Auth Code",
    key="edit_job_search",
)

search_col.caption(
    "Search uses the API directly, so older jobs can still be found without loading every record."
)

try:
    df = load_jobs_df(job_search).copy()
except requests.exceptions.ConnectionError:
    st.error(f"Could not connect to API at {API_URL}")
    st.stop()
except requests.exceptions.Timeout:
    st.error("API request timed out while loading jobs.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load jobs from API: {e}")
    st.stop()

if df.empty:
    if job_search.strip():
        st.warning("No matching jobs found.")
    else:
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


def label_row(r) -> str:
    jid = str(r.get("job_id", "") or "")
    vreg = str(r.get("vehicle_reg", "") or "")
    wdt = r.get("work_date", None)
    d = wdt.isoformat() if hasattr(wdt, "isoformat") else str(wdt or "")
    return f"#{int(r['id'])} | {jid} | {vreg} | {d}"


rows = df.sort_values(by="id", ascending=False).to_dict("records")
labels = [label_row(r) for r in rows]
options = [None] + list(range(len(rows)))

if st.session_state.edit_job_choice not in options:
    st.session_state.edit_job_choice = None

choice = select_col.selectbox(
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

if st.session_state.edit_expense_row_id != row_id:
    st.session_state.edit_expense_rows = parse_expenses_for_edit(
        job.get("job_expenses"),
        job.get("expenses_amount") or 0.0,
    )
    st.session_state.edit_expense_row_id = row_id

st.info(
    f"Editing Row #{row_id} | "
    f"Job: {job.get('job_id') or ''} | "
    f"Vehicle: {job.get('vehicle_reg') or ''}"
)

left_col, right_col = st.columns([2.2, 1])

with left_col:
    with st.container(border=True):
        st.markdown("### 📋 Job Information")

        col1, col2, col3 = st.columns(3)

        work_date = col1.date_input(
            "Date",
            value=job.get("work_date") or date.today(),
        )

        job_number = col2.text_input(
            "Job Number",
            value=str(job.get("job_id") or ""),
        )

        job_type = col3.selectbox(
            "Job Type",
            cfg.JOB_TYPE_OPTIONS,
            index=(
                cfg.JOB_TYPE_OPTIONS.index(job.get("category"))
                if job.get("category") in cfg.JOB_TYPE_OPTIONS
                else 0
            ),
        )

        col4, col5 = st.columns(2)

        current_outcome = str(job.get("job_outcome") or "Completed")
        if current_outcome not in OUTCOME_OPTIONS:
            current_outcome = "Completed"

        job_outcome = col4.selectbox(
            "Job Outcome",
            OUTCOME_OPTIONS,
            index=OUTCOME_OPTIONS.index(current_outcome),
        )

        current_status = str(job.get(STATUS_COL) or "Pending") if STATUS_COL else "Pending"

        job_status = col5.selectbox(
            "Job Status",
            cfg.STATUS_OPTIONS,
            index=(
                cfg.STATUS_OPTIONS.index(current_status)
                if current_status in cfg.STATUS_OPTIONS
                else 0
            ),
        )

    with st.container(border=True):
        st.markdown("### 🚗 Vehicle Details")

        col1, col2 = st.columns(2)

        vehicle_description = col1.text_input(
            "Vehicle Description",
            value=str(job.get("vehicle_description") or ""),
        )

        vehicle_reg = col2.text_input(
            "Vehicle Reg",
            value=str(job.get("vehicle_reg") or ""),
        )

    with st.container(border=True):
        st.markdown("### 📍 Journey")

        col1, col2 = st.columns(2)

        collection_from = col1.text_input(
            "Collection From",
            value=str(job.get("collection_from") or ""),
        )

        delivery_to = col2.text_input(
            "Delivery To",
            value=str(job.get("delivery_to") or ""),
        )

    with st.container(border=True):
        st.markdown("### 💰 Job Pay & Expenses")

        job_amount = st.number_input(
            "Job Amount (£)",
            min_value=0.0,
            step=1.0,
            value=float(job.get("amount") or 0.0),
        )

        st.markdown("#### Expenses")

        for i, expense in enumerate(st.session_state.edit_expense_rows):
            exp_col1, exp_col2, exp_col3 = st.columns([2, 1, 0.7])

            expense_type = exp_col1.selectbox(
                f"Expense Type {i + 1}",
                cfg.JOB_EXPENSE_OPTIONS,
                index=(
                    cfg.JOB_EXPENSE_OPTIONS.index(expense.get("type"))
                    if expense.get("type") in cfg.JOB_EXPENSE_OPTIONS
                    else 0
                ),
                key=f"edit_expense_type_{row_id}_{i}",
            )

            expense_amount = exp_col2.number_input(
                f"Amount {i + 1} (£)",
                min_value=0.0,
                step=0.5,
                value=float(expense.get("amount") or 0.0),
                key=f"edit_expense_amount_{row_id}_{i}",
            )

            st.session_state.edit_expense_rows[i] = {
                "type": expense_type,
                "amount": float(expense_amount),
            }

            with exp_col3:
                st.write("")
                st.write("")
                if len(st.session_state.edit_expense_rows) > 1:
                    if st.button("Remove", key=f"edit_remove_expense_{row_id}_{i}"):
                        st.session_state.edit_expense_rows.pop(i)
                        st.rerun()

        if st.button("➕ Add Another Expense", key=f"edit_add_expense_{row_id}"):
            st.session_state.edit_expense_rows.append(
                {"type": "No expenses", "amount": 0.0}
            )
            st.rerun()

    job_expenses, expenses_amount = build_expenses_from_rows(
        st.session_state.edit_expense_rows
    )

    with st.container(border=True):
        st.markdown("### ⏱ Waiting Time")

        col1, col2, col3 = st.columns(3)

        waiting_time = col1.text_input(
            "Waiting Time",
            value=str(job.get("waiting_time") or ""),
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

        add_pay_value = float(job.get("add_pay") or 0.0) if "add_pay" in df.columns else 0.0

        add_pay = col1.number_input(
            "Add Pay (£)",
            min_value=0.0,
            step=1.0,
            value=add_pay_value,
        )

        hours = col2.number_input(
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

        paid_date = col3.date_input(
            "Paid Date",
            value=default_paid_date,
            disabled=(("paid_date" not in df.columns) or (str(job_status).strip().lower() != "paid")),
            help=(
                "Auto-fills to today when status is Paid, but you can change it manually."
                if "paid_date" in df.columns
                else "DB/API does not currently expose a paid_date field."
            ),
        )

        auth_code = st.text_input(
            "Auth Code",
            value=str(job.get("auth_code") or ""),
        )

        comments = st.text_area(
            "Comments",
            value=str(job.get("comments") or ""),
        )


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

        if st.session_state.edit_expense_rows:
            valid_expense_rows = [
                row
                for row in st.session_state.edit_expense_rows
                if row.get("type") != "No expenses"
                and float(row.get("amount") or 0.0) > 0
            ]

            if valid_expense_rows:
                st.markdown("#### Expense Breakdown")
                for row in valid_expense_rows:
                    st.write(f"- {row['type']}: £{float(row['amount']):.2f}")
            else:
                st.caption("No expenses added.")
        else:
            st.caption("No expenses added.")

    submitted = st.button(
        "💾 Update Job",
        type="primary",
        use_container_width=True,
    )


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

        old_is_na = pd.isna(old_cmp) if not isinstance(old_cmp, (list, dict)) else False
        new_is_na = pd.isna(new_cmp) if not isinstance(new_cmp, (list, dict)) else False

        if old_is_na and new_is_na:
            return

        if old_cmp == new_cmp:
            return

        diffs[db_col] = new_val

    set_if_changed("work_date", work_date.isoformat() if work_date else None)
    set_if_changed("job_id", clean_text(job_number))
    set_if_changed("category", job_type)
    set_if_changed("vehicle_description", clean_text(vehicle_description))

    cleaned_reg = clean_text(vehicle_reg)
    set_if_changed("vehicle_reg", cleaned_reg.upper() if cleaned_reg else None)

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
            set_if_changed(
                "paid_date",
                paid_date.isoformat() if paid_date else date.today().isoformat(),
            )
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

with st.container(border=True):
    st.markdown("### 🗑 Danger Zone")

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