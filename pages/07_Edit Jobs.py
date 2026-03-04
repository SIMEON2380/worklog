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

# Optional filters (safe even if columns missing)
c1, c2, c3 = st.columns(3)

job_id_filter = c1.text_input("Filter by Job Number (optional)") if "job_id" in df.columns else ""
status_filter = c2.selectbox("Filter by Status (optional)", ["All"] + sorted(df["status"].dropna().astype(str).unique().tolist())) if "status" in df.columns else "All"

if "work_date" in df.columns:
    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce").dt.date
    min_d = df["work_date"].dropna().min()
    max_d = df["work_date"].dropna().max()
    date_range = c3.date_input("Filter by Date Range (optional)", value=(min_d, max_d))
else:
    date_range = None

filtered = df.copy()

if job_id_filter and "job_id" in filtered.columns:
    filtered = filtered[filtered["job_id"].astype(str).str.contains(job_id_filter.strip(), na=False)]

if status_filter != "All" and "status" in filtered.columns:
    filtered = filtered[filtered["status"].astype(str) == status_filter]

if date_range and "work_date" in filtered.columns and isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered["work_date"] >= start) & (filtered["work_date"] <= end)]

st.caption("Edit values in the table, then click **Save changes**.")

# Choose which columns are editable (keep this tight to avoid messing up schema)
editable_cols = []
for col in ["status", "amount", "waiting_hours", "waiting_amount", "expenses_amount", "comment", "comments", "note", "notes"]:
    if col in filtered.columns:
        editable_cols.append(col)

if not editable_cols:
    st.warning("No editable columns found (expected columns like status/amount/etc).")
    st.dataframe(filtered)
    st.stop()

# Build editor view: keep id visible (or hide if you want, but we NEED it present)
editor_df = filtered[["id"] + editable_cols].copy()

edited = st.data_editor(
    editor_df,
    use_container_width=True,
    num_rows="fixed",
    key="edit_jobs_editor",
)

# Save button
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

            diffs[col] = None if (isinstance(new, float) and pd.isna(new)) else new

        if diffs:
            DB["update_row"](int(row_id), diffs)
            changes += 1

    if changes:
        st.success(f"Saved changes for {changes} row(s).")
        st.rerun()
    else:
        st.info("No changes to save.")