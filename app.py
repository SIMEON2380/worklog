import streamlit as st
import pandas as pd

from worklog.constants import (
    UI_COLUMNS,
    JOB_TYPE_OPTIONS,
    STATUS_OPTIONS,
    EXPENSE_TYPE_OPTIONS,
)
from worklog.db import (
    init_db,
    upsert_job,
    list_jobs,
    get_job_by_number,
    update_job,
    delete_job,
)
from worklog.normalize import (
    clean_job_number,
    clean_text,
    clean_postcode,
    normalize_job_type,
    normalize_status,
    normalize_expense_type,
)

st.set_page_config(page_title="Worklog", layout="wide")

# --- boot ---
init_db()

st.title("Worklog")

tab_add, tab_view, tab_edit = st.tabs(["Add / Upsert", "View", "Edit by Job Number"])

# ----------------------------
# ADD / UPSERT
# ----------------------------
with tab_add:
    st.subheader("Add / Upsert Job")

    with st.form("add_job_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            job_number = st.text_input("Job Number")
            job_type = st.selectbox("Job Type", JOB_TYPE_OPTIONS, key="add_job_type")
            status = st.selectbox("Status", STATUS_OPTIONS, key="add_status")

        with c2:
            vehicle_description = st.text_input("Vehicle Description")
            postcode = st.text_input("Postcode")
            expense_type = st.selectbox("Expense Type", EXPENSE_TYPE_OPTIONS, key="add_expense_type")

        with c3:
            customer_name = st.text_input("Customer Name")
            site_address = st.text_input("Site Address")
            notes = st.text_area("Notes", height=110)

        submitted = st.form_submit_button("Save")

        if submitted:
            job_number_n = clean_job_number(job_number)
            if not job_number_n:
                st.error("Job Number is required.")
            else:
                record = {
                    "job_number": job_number_n,
                    "job_type": normalize_job_type(job_type),
                    "status": normalize_status(status),
                    "vehicle_description": clean_text(vehicle_description),
                    "postcode": clean_postcode(postcode),
                    "expense_type": normalize_expense_type(expense_type),
                    "customer_name": clean_text(customer_name),
                    "site_address": clean_text(site_address),
                    "notes": clean_text(notes),
                }
                upsert_job(record)
                st.success(f"Saved job {job_number_n}")

# ----------------------------
# VIEW
# ----------------------------
with tab_view:
    st.subheader("All Jobs")

    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        search = st.text_input("Search (job number / postcode / vehicle / customer)", key="view_search")
    with colB:
        limit = st.number_input("Max rows", min_value=50, max_value=5000, value=500, step=50)
    with colC:
        refresh = st.button("Refresh")

    df = list_jobs(search=search, limit=int(limit))
    if df.empty:
        st.info("No jobs found.")
    else:
        # show only known columns that exist (so app won't crash if db schema differs)
        cols = [c for c in UI_COLUMNS if c in df.columns]
        if not cols:
            cols = df.columns.tolist()
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Delete Job (by Job Number)")

    del_job = st.text_input("Job Number to delete", key="delete_job_number")
    if st.button("Delete", type="primary"):
        del_job_n = clean_job_number(del_job)
        if not del_job_n:
            st.error("Enter a job number.")
        else:
            ok = delete_job(del_job_n)
            if ok:
                st.success(f"Deleted {del_job_n}")
            else:
                st.warning("Job number not found.")

# ----------------------------
# EDIT BY JOB NUMBER
# ----------------------------
with tab_edit:
    st.subheader("Edit Job by Job Number")

    job_number_input = st.text_input("Enter Job Number", key="edit_lookup_job_number")

    if job_number_input:
        job_number_n = clean_job_number(job_number_input)
        if not job_number_n:
            st.error("Invalid job number.")
        else:
            row = get_job_by_number(job_number_n)

            if row is None:
                st.warning("Job number not found.")
            else:
                st.caption(f"Editing job: **{job_number_n}**")

                # Build safe indexes (prevents crashes if DB has unexpected values)
                def safe_index(options, value):
                    return options.index(value) if value in options else 0

                with st.form("edit_job_form"):
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        job_type = st.selectbox(
                            "Job Type",
                            JOB_TYPE_OPTIONS,
                            index=safe_index(JOB_TYPE_OPTIONS, row.get("job_type", "")),
                            key="edit_job_type",
                        )
                        status = st.selectbox(
                            "Status",
                            STATUS_OPTIONS,
                            index=safe_index(STATUS_OPTIONS, row.get("status", "")),
                            key="edit_status",
                        )

                    with c2:
                        vehicle_description = st.text_input(
                            "Vehicle Description",
                            value=row.get("vehicle_description", "") or "",
                            key="edit_vehicle_description",
                        )
                        postcode = st.text_input(
                            "Postcode",
                            value=row.get("postcode", "") or "",
                            key="edit_postcode",
                        )

                    with c3:
                        expense_type = st.selectbox(
                            "Expense Type",
                            EXPENSE_TYPE_OPTIONS,
                            index=safe_index(EXPENSE_TYPE_OPTIONS, row.get("expense_type", "")),
                            key="edit_expense_type",
                        )
                        customer_name = st.text_input(
                            "Customer Name",
                            value=row.get("customer_name", "") or "",
                            key="edit_customer_name",
                        )

                    site_address = st.text_input(
                        "Site Address",
                        value=row.get("site_address", "") or "",
                        key="edit_site_address",
                    )
                    notes = st.text_area(
                        "Notes",
                        value=row.get("notes", "") or "",
                        height=120,
                        key="edit_notes",
                    )

                    submitted = st.form_submit_button("Update Job")

                    if submitted:
                        update_job(
                            job_number=job_number_n,
                            job_type=normalize_job_type(job_type),
                            status=normalize_status(status),
                            vehicle_description=clean_text(vehicle_description),
                            postcode=clean_postcode(postcode),
                            expense_type=normalize_expense_type(expense_type),
                            customer_name=clean_text(customer_name),
                            site_address=clean_text(site_address),
                            notes=clean_text(notes),
                        )
                        st.success("Job updated.")