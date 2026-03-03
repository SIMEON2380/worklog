from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from .config import Config


# -------------------------
# Login gate
# -------------------------
def require_login() -> None:
    if not st.session_state.get("auth_user"):
        st.warning("Please log in from the Dashboard page.")
        st.stop()


# -------------------------
# UI <-> DB mapping
# -------------------------
def ui_to_db_map(cfg: Config) -> Dict[str, str]:
    # keep labels exactly as in cfg.UI_COLUMNS (including typos)
    return {
        "Date": "work_date",
        "job number": "job_id",
        "job type": "category",
        "vehcile description": "vehicle_description",
        "vehicle Reg": "vehicle_reg",
        "collection from": "collection_from",
        "delivery to": "delivery_to",
        "job amount": "amount",
        "Job Expenses": "job_expenses",
        "expenses Amount": "expenses_amount",
        "Auth code": "auth_code",
        "job status": "job_status",
        "waiting time": "waiting_time",
        "comments": "comments",
    }


def db_to_ui_map(cfg: Config) -> Dict[str, str]:
    m = ui_to_db_map(cfg)
    return {v: k for k, v in m.items()}


def to_ui_table(cfg: Config, df_db: pd.DataFrame) -> pd.DataFrame:
    """
    UI-formatted dataframe:
    - keep id (hidden in display, used for edits)
    - columns in cfg.UI_COLUMNS order + exact labels
    """
    if df_db is None or df_db.empty:
        cols = ["id"] + list(cfg.UI_COLUMNS)
        return pd.DataFrame(columns=cols)

    df = df_db.copy()

    # Ensure id exists
    if "id" not in df.columns:
        df["id"] = None

    # Ensure expected DB cols exist
    for c in cfg.EXPECTED_DB_COLS:
        if c not in df.columns:
            df[c] = None

    m = ui_to_db_map(cfg)

    out = pd.DataFrame()
    out["id"] = df["id"]

    for ui_col in cfg.UI_COLUMNS:
        db_col = m.get(ui_col)
        out[ui_col] = df[db_col] if db_col in df.columns else None

    return out


def ui_row_to_db_fields(cfg: Config, ui_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a row dict from UI labels to DB fields.
    """
    m = ui_to_db_map(cfg)
    out: Dict[str, Any] = {}
    for ui_col, db_col in m.items():
        if ui_col in ui_row:
            out[db_col] = ui_row[ui_col]
    return out


# -------------------------
# Totals (reports)
# -------------------------
def compute_totals(df_db: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Backwards-compatible totals:
      total_job_amount, total_wait_hours, total_wait_amount
    """
    if df_db is None or df_db.empty:
        return 0.0, 0.0, 0.0

    amt = float(pd.to_numeric(df_db.get("amount"), errors="coerce").fillna(0).sum())
    wh = float(pd.to_numeric(df_db.get("waiting_hours"), errors="coerce").fillna(0).sum())
    wa = float(pd.to_numeric(df_db.get("waiting_amount"), errors="coerce").fillna(0).sum())
    return amt, wh, wa


def show_totals(df_db: pd.DataFrame) -> None:
    """
    Kept exactly so existing pages/features don't break.
    """
    total_job_amount, total_wait_hours, total_wait_amount = compute_totals(df_db)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total job amount", f"£{total_job_amount:,.2f}")
    c2.metric("Total waiting time", f"{total_wait_hours:,.2f} hrs")
    c3.metric("Waiting total", f"£{total_wait_amount:,.2f}")


def report_totals(df: pd.DataFrame) -> Dict[str, float]:
    """
    Robust totals for reports (used by Daily/Weekly/Monthly):
      job_amount, wait_hours, wait_pay, expenses, grand_total
    grand_total = job_amount + wait_pay - expenses
    """
    if df is None or df.empty:
        return {
            "job_amount": 0.0,
            "wait_hours": 0.0,
            "wait_pay": 0.0,
            "expenses": 0.0,
            "grand_total": 0.0,
        }

    def num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([0.0] * len(df), index=df.index)
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    job_amount = float(num("amount").sum())
    wait_hours = float(num("waiting_hours").sum())
    wait_pay = float(num("waiting_amount").sum())
    expenses = float(num("expenses_amount").sum())
    grand_total = job_amount + wait_pay - expenses

    return {
        "job_amount": job_amount,
        "wait_hours": wait_hours,
        "wait_pay": wait_pay,
        "expenses": expenses,
        "grand_total": grand_total,
    }


# -------------------------
# Table display (non-edit)
# -------------------------
def display_jobs_table(cfg: Config, df_db: pd.DataFrame, caption: Optional[str] = None) -> None:
    if caption:
        st.caption(caption)

    ui_df = to_ui_table(cfg, df_db)
    show_df = ui_df.drop(columns=["id"], errors="ignore")
    st.dataframe(show_df, use_container_width=True)


# -------------------------
# Editable table (save back to DB)
# -------------------------
def editable_jobs_table(
    cfg: Config,
    DB: Dict[str, Any],
    df_db: pd.DataFrame,
    key: str,
    allow_type_edit: bool = True,
) -> None:
    """
    Inline editor + Save button using your db.py API: update_row_by_id().
    """
    if df_db is None or df_db.empty:
        st.info("No rows to show.")
        return

    ui_df = to_ui_table(cfg, df_db)

    disabled_cols = ["id"]
    if not allow_type_edit:
        disabled_cols.append("job type")

    edited = st.data_editor(
        ui_df,
        key=key,
        num_rows="fixed",
        disabled=disabled_cols,
        column_config={
            "job status": st.column_config.SelectboxColumn("job status", options=cfg.STATUS_OPTIONS),
            "job type": st.column_config.SelectboxColumn("job type", options=cfg.JOB_TYPE_OPTIONS),
            "Job Expenses": st.column_config.SelectboxColumn("Job Expenses", options=cfg.JOB_EXPENSE_OPTIONS),
        },
        use_container_width=True,
    )

    if st.button("Save changes", key=f"{key}_save"):
        original = ui_df.set_index("id")
        updated = edited.set_index("id")
        changes = 0

        for row_id in updated.index:
            before = original.loc[row_id].to_dict()
            after = updated.loc[row_id].to_dict()

            # detect diffs (UI-space)
            changed = False
            for k, v in after.items():
                b = before.get(k)
                if pd.isna(b) and pd.isna(v):
                    continue
                if (pd.isna(b) and not pd.isna(v)) or (not pd.isna(b) and pd.isna(v)) or (str(b) != str(v)):
                    changed = True
                    break

            if not changed:
                continue

            row_db = ui_row_to_db_fields(cfg, after)

            # Convert date safely
            wd = row_db.get("work_date")
            wd_dt = pd.to_datetime(wd, errors="coerce").date() if wd not in (None, "") else date.today()

            # Convert numbers safely
            def fnum(x):
                try:
                    return float(x)
                except Exception:
                    return None

            DB["update_row_by_id"](
                int(row_id),
                wd_dt,
                str(row_db.get("job_id") or ""),
                str(row_db.get("category") or cfg.JOB_TYPE_OPTIONS[0]),
                str(row_db.get("vehicle_description") or ""),
                str(row_db.get("vehicle_reg") or ""),
                str(row_db.get("collection_from") or ""),
                str(row_db.get("delivery_to") or ""),
                fnum(row_db.get("amount")),
                row_db.get("job_expenses"),
                fnum(row_db.get("expenses_amount")),
                str(row_db.get("auth_code") or ""),
                str(row_db.get("job_status") or "Pending"),
                str(row_db.get("waiting_time") or ""),
                str(row_db.get("comments") or ""),
            )

            changes += 1

        st.success(f"Saved changes for {changes} job(s).")
        st.rerun()