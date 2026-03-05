import streamlit as st
import pandas as pd
from datetime import date

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Edit Jobs", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Edit Jobs (Form)")

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


df = DB["read_all"]().copy()
if df.empty:
    st.info("No jobs found.")
    st.stop()

if "id" not in df.columns:
    st.error("Missing 'id' column in dataset. Cannot edit safely.")
    st.stop()

# Normalize dates for UI
if "work_date" in df.columns:
    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date

# Prefer job_status if present, else status
STATUS_COL = "job_status" if "job_status" in df.columns else ("status" if "status" in df.columns else None)

# -------------------------
# Pick a job
# -------------------------
left, right = st.columns([2, 3])

job_search = left.text_input("Search by Job Number / Vehicle Reg (optional)")
filtered = df.copy()

if job_search.strip():
    s = job_search.strip().lower()
    mask = False
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

choice = right.selectbox(
    "Select job to edit",
    options=list(range(len(rows))),
    format_func=lambda i: labels[i],
)

job = rows[choice]
row_id = int(job["id"])

st.divider()

# -------------------------
# Edit form (same style as Add Entry)
# -------------------------
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

    current_status = str(job.get(STATUS_COL) or "Start") if STATUS_COL else "Start"
    job_status = col6.selectbox(
        "Job Status",
        cfg.STATUS_OPTIONS,
        index=(cfg.STATUS_OPTIONS.index(current_status) if current_status in cfg.STATUS_OPTIONS else 0),
    )

    col7, col8, col9 = st.columns(3)
    collection_from = col7.text_input("Collection From", value=str(job.get("collection_from") or ""))
    delivery_to = col8.text_input("Delivery To", value=str(job.get("delivery_to") or ""))
    job_amount = col9.number_input(
        "Job Amount (£)",
        min_value=0.0,
        step=1.0,
        value=float(job.get("amount") or 0.0),
    )

    col10, col11, col12 = st.columns(3)
    job_expenses = col10.selectbox(
        "Job Expenses",
        cfg.JOB_EXPENSE_OPTIONS,
        index=(cfg.JOB_EXPENSE_OPTIONS.index(job.get("job_expenses")) if job.get("job_expenses") in cfg.JOB_EXPENSE_OPTIONS else 0),
    )
    expenses_amount = col11.number_input(
        "Expenses Amount (£)",
        min_value=0.0,
        step=0.5,
        value=float(job.get("expenses_amount") or 0.0),
    )

    # --- WAITING TIME: edit in the same format your reports expect ---
    waiting_time = col12.text_input(
        "Waiting Time (e.g. 10-11 or 09:00-11:30)",
        value=str(job.get("waiting_time") or ""),
    )

    # Auto-derived values from waiting_time
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

    hours = col15.number_input(
        "Hours (if used)",
        min_value=0.0,
        step=0.5,
        value=float(job.get("hours") or 0.0),
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
        set_if_changed("job_id", job_number.strip() if job_number else None)
        set_if_changed("category", job_type)
        set_if_changed("vehicle_description", vehicle_description.strip() if vehicle_description else None)
        set_if_changed("vehicle_reg", vehicle_reg.strip() if vehicle_reg else None)
        set_if_changed("collection_from", collection_from.strip() if collection_from else None)
        set_if_changed("delivery_to", delivery_to.strip() if delivery_to else None)

        set_if_changed("amount", float(job_amount))
        set_if_changed("job_expenses", job_expenses)
        set_if_changed("expenses_amount", float(expenses_amount))

        # Save waiting in the format reports expect + keep numeric columns aligned
        set_if_changed("waiting_time", waiting_time.strip() if waiting_time else None)
        set_if_changed("waiting_hours", float(calc_waiting_hours))
        set_if_changed("waiting_amount", float(calc_waiting_amount))

        set_if_changed("hours", float(hours))

        set_if_changed("auth_code", auth_code.strip() if auth_code else None)
        set_if_changed("comments", comments.strip() if comments else None)

        # Status: keep both columns in sync if both exist
        if STATUS_COL:
            set_if_changed(STATUS_COL, job_status)
            if STATUS_COL == "job_status" and "status" in df.columns:
                set_if_changed("status", job_status)
            if STATUS_COL == "status" and "job_status" in df.columns:
                set_if_changed("job_status", job_status)

        if diffs:
            DB["update_row"](row_id, diffs)
            st.success("Saved. Reports will reflect this immediately.")
            st.rerun()
        else:
            st.info("No changes detected.")