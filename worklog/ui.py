from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import streamlit as st

from .config import Config
from .db import fetch_jobs, insert_job, update_job_fields, update_job_status


def require_login() -> None:
    if not st.session_state.get("auth_user"):
        st.stop()


def show_top_metrics(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No jobs found for the selected filters.")
        return

    total_jobs = len(df)
    total_pay = float(df["pay"].sum())
    total_exp = float(df["expense_amount"].sum())
    net = total_pay - total_exp

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs", f"{total_jobs}")
    c2.metric("Pay", f"£{total_pay:,.2f}")
    c3.metric("Expenses", f"£{total_exp:,.2f}")
    c4.metric("Net", f"£{net:,.2f}")


def status_bar(df: pd.DataFrame, cfg: Config) -> None:
    if df.empty:
        return

    counts = df["status"].value_counts().reindex(cfg.STATUS_OPTIONS, fill_value=0)
    bar_df = pd.DataFrame({"status": counts.index, "count": counts.values})
    bar_df = bar_df[bar_df["count"] > 0]

    if bar_df.empty:
        return

    st.subheader("Status overview")
    st.bar_chart(bar_df.set_index("status"))


def job_entry_form(cfg: Config) -> None:
    with st.expander("➕ Add new job", expanded=False):
        col1, col2, col3 = st.columns(3)

        job_date = col1.date_input("Job date")
        job_type = col2.selectbox("Job type", cfg.JOB_TYPE_OPTIONS)
        status = col3.selectbox("Status", cfg.STATUS_OPTIONS, index=0)

        col4, col5, col6 = st.columns(3)
        reference = col4.text_input("Reference (optional)")
        start_time = col5.text_input("Start time (optional)", placeholder="e.g. 09:00")
        end_time = col6.text_input("End time (optional)", placeholder="e.g. 12:30")

        col7, col8, col9 = st.columns(3)
        waiting_hours = col7.number_input("Waiting hours", min_value=0.0, step=0.25, value=0.0)
        pay = col8.number_input("Pay (£)", min_value=0.0, step=1.0, value=0.0)
        expense_type = col9.selectbox("Expense type", [""] + cfg.JOB_EXPENSE_OPTIONS)

        expense_amount = st.number_input("Expense amount (£)", min_value=0.0, step=0.5, value=0.0)
        notes = st.text_area("Notes (optional)", height=80)

        if st.button("Save job", type="primary"):
            insert_job(cfg, {
                "job_date": str(job_date),
                "job_type": job_type,
                "status": status,
                "reference": reference,
                "start_time": start_time,
                "end_time": end_time,
                "waiting_hours": waiting_hours,
                "pay": pay,
                "expense_type": expense_type,
                "expense_amount": expense_amount,
                "notes": notes,
            })
            st.success("Job saved.")
            st.rerun()


def editable_jobs_table(
    cfg: Config,
    df: pd.DataFrame,
    key: str,
    allow_type_edit: bool = True,
) -> None:
    """
    Inline editor with Save button.
    """
    if df.empty:
        st.write("No rows to show.")
        return

    show = df.copy()

    # Keep id visible but not editable
    disabled = ["id", "created_at", "updated_at"]
    if not allow_type_edit:
        disabled.append("job_type")

    edited = st.data_editor(
        show,
        key=key,
        num_rows="fixed",
        disabled=disabled,
        column_config={
            "status": st.column_config.SelectboxColumn("status", options=cfg.STATUS_OPTIONS),
            "job_type": st.column_config.SelectboxColumn("job_type", options=cfg.JOB_TYPE_OPTIONS),
            "expense_type": st.column_config.SelectboxColumn("expense_type", options=[""] + cfg.JOB_EXPENSE_OPTIONS),
        },
        use_container_width=True,
    )

    # Save changes
    if st.button("Save changes", key=f"{key}_save"):
        # Compare row by row
        changes = 0
        original = df.set_index("id")
        updated = edited.set_index("id")

        for job_id in updated.index:
            before = original.loc[job_id].to_dict()
            after = updated.loc[job_id].to_dict()

            diff = {}
            for k, v in after.items():
                if k in ("created_at", "updated_at"):
                    continue
                # Normalize nan
                b = before.get(k)
                if pd.isna(b) and pd.isna(v):
                    continue
                if (pd.isna(b) and not pd.isna(v)) or (not pd.isna(b) and pd.isna(v)) or (str(b) != str(v)):
                    diff[k] = v

            if diff:
                update_job_fields(cfg, int(job_id), diff)
                changes += 1

        st.success(f"Saved changes for {changes} job(s).")
        st.rerun()