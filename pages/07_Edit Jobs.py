import streamlit as st
import pandas as pd

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

st.subheader("Edit Jobs")

df = DB["read_all"]().copy()
if df.empty:
    st.info("No jobs found.")
    st.stop()

# Make sure id exists (required for saving)
if "id" not in df.columns:
    st.error("Missing 'id' column in dataset. Cannot safely save edits.")
    st.stop()

# --- Choose the status column to edit ---
# Your DB has BOTH 'job_status' and 'status'. Prefer 'job_status'.
STATUS_COL = "job_status" if "job_status" in df.columns else ("status" if "status" in df.columns else None)

# --- Choose job type column ---
# In your schema, job type is stored as 'category'
JOB_TYPE_COL = "category" if "category" in df.columns else None

# Optional filters
c1, c2, c3, c4 = st.columns(4)

job_id_filter = c1.text_input("Filter by Job Number (optional)") if "job_id" in df.columns else ""

status_filter = "All"
if STATUS_COL:
    existing_statuses = (
        df[STATUS_COL].dropna().astype(str).unique().tolist()
        if not df[STATUS_COL].dropna().empty
        else []
    )
    status_options = ["All"] + list(dict.fromkeys(cfg.STATUS_OPTIONS + sorted(existing_statuses)))
    status_filter = c2.selectbox("Filter by Status (optional)", status_options)
else:
    c2.info("No status column found.")

job_type_filter = "All"
if JOB_TYPE_COL:
    existing_types = (
        df[JOB_TYPE_COL].dropna().astype(str).unique().tolist()
        if not df[JOB_TYPE_COL].dropna().empty
        else []
    )
    type_options = ["All"] + list(dict.fromkeys(cfg.JOB_TYPE_OPTIONS + sorted(existing_types)))
    job_type_filter = c4.selectbox("Filter by Job Type (optional)", type_options)
else:
    c4.info("No job type column found.")

date_range = None
if "work_date" in df.columns:
    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date
    min_d = df["work_date"].dropna().min()
    max_d = df["work_date"].dropna().max()
    if pd.notna(min_d) and pd.notna(max_d):
        date_range = c3.date_input("Filter by Date Range (optional)", value=(min_d, max_d))
else:
    c3.info("No work_date column found.")

filtered = df.copy()

if job_id_filter and "job_id" in filtered.columns:
    filtered = filtered[filtered["job_id"].astype(str).str.contains(job_id_filter.strip(), na=False)]

if status_filter != "All" and STATUS_COL and STATUS_COL in filtered.columns:
    filtered = filtered[filtered[STATUS_COL].astype(str) == status_filter]

if job_type_filter != "All" and JOB_TYPE_COL and JOB_TYPE_COL in filtered.columns:
    filtered = filtered[filtered[JOB_TYPE_COL].astype(str) == job_type_filter]

if (
    date_range
    and "work_date" in filtered.columns
    and isinstance(date_range, tuple)
    and len(date_range) == 2
):
    start, end = date_range
    filtered = filtered[(filtered["work_date"] >= start) & (filtered["work_date"] <= end)]

st.caption("Edit values in the table, then click **Save changes**.")

# Editable columns (keep tight)
editable_cols = []

# job type first (so you can edit it)
if JOB_TYPE_COL and JOB_TYPE_COL in filtered.columns:
    editable_cols.append(JOB_TYPE_COL)

# status
if STATUS_COL and STATUS_COL in filtered.columns:
    editable_cols.append(STATUS_COL)

# other columns that actually exist in your schema
for col in [
    "amount",
    "waiting_hours",
    "waiting_amount",
    "expenses_amount",
    "comments",  # ✅ your DB column is 'comments'
]:
    if col in filtered.columns:
        editable_cols.append(col)

if not editable_cols:
    st.warning("No editable columns found.")
    st.dataframe(filtered, use_container_width=True)
    st.stop()

editor_df = filtered[["id"] + editable_cols].copy()

# Column configs (dropdowns)
column_config = {}

if JOB_TYPE_COL:
    column_config[JOB_TYPE_COL] = st.column_config.SelectboxColumn(
        "Job Type",
        options=cfg.JOB_TYPE_OPTIONS,
        help="Change job type",
        required=False,
    )

if STATUS_COL:
    column_config[STATUS_COL] = st.column_config.SelectboxColumn(
        "Job Status",
        options=cfg.STATUS_OPTIONS,
        help="Change job status",
        required=False,
    )

edited = st.data_editor(
    editor_df,
    use_container_width=True,
    num_rows="fixed",
    key="edit_jobs_editor",
    column_config=column_config if column_config else None,
)

# Save changes
if st.button("Save changes", type="primary"):
    original = editor_df.set_index("id")
    updated = edited.set_index("id")

    changes = 0
    for row_id in updated.index:
        if row_id not in original.index:
            continue

        diffs = {}
        for col in editable_cols:
            old = original.at[row_id, col]
            new = updated.at[row_id, col]

            # Treat NaN == NaN as no change
            if (pd.isna(old) and pd.isna(new)) or old == new:
                continue

            # Normalize blanks in status
            if col == STATUS_COL:
                if new is None or (isinstance(new, str) and not new.strip()):
                    new = "Start"

            diffs[col] = None if (isinstance(new, float) and pd.isna(new)) else new

            # ✅ Keep both columns in sync if both exist
            if col == "job_status" and "status" in df.columns:
                diffs["status"] = new
            if col == "status" and "job_status" in df.columns:
                diffs["job_status"] = new

        if diffs:
            DB["update_row"](int(row_id), diffs)
            changes += 1

    if changes:
        st.success(f"Saved changes for {changes} row(s).")
        st.rerun()
    else:
        st.info("No changes to save.")