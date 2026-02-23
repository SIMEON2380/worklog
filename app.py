# app.py
# Streamlit Worklog App (SQLite) — full rewrite with ALL requested rules + previous core features preserved
# Key updates in this version:
# - EXCLUDE Withdraw from £8 "Inspect & Collect" total
# - After save: clear input fields EXCEPT date
# - Strong validation: stop wrong-field input (e.g., job number in vehicle description)
# - Vehicle description always stored/displayed in UPPERCASE
# - Withdraw is excluded from "total amount of work" (overall totals)
# - Comment column included
# - Dashboard can DISPLAY Withdraw/Aborted jobs (but totals can exclude withdraw where needed)
# - Always fetch latest data (no caching)

import re
import io
import sqlite3
from datetime import datetime, date
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

APP_VERSION = "v2.6.0"
DB_PATH = "worklog.db"

# ----------------------------
# Page / App settings
# ----------------------------
st.set_page_config(page_title="Worklog", layout="wide")
st.title("Worklog")
st.caption(f"App version: {APP_VERSION}")

# ----------------------------
# DB Helpers
# ----------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    # Base table (id + created_at used to keep data stable and editable)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS worklog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,

            work_date TEXT NOT NULL,
            job_number TEXT NOT NULL,

            job_type TEXT NOT NULL,
            job_status TEXT NOT NULL,

            vehicle_description TEXT NOT NULL,
            amount REAL NOT NULL DEFAULT 0.0,
            waiting_time TEXT NOT NULL DEFAULT '',
            comment TEXT NOT NULL DEFAULT ''
        )
        """
    )

    # Add any missing columns safely (migrations)
    existing_cols = {r[1] for r in cur.execute("PRAGMA table_info(worklog)").fetchall()}
    migrations = [
        ("comment", "ALTER TABLE worklog ADD COLUMN comment TEXT NOT NULL DEFAULT ''"),
    ]
    for col, sql in migrations:
        if col not in existing_cols:
            cur.execute(sql)

    # Useful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_worklog_date ON worklog(work_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_worklog_job ON worklog(job_number)")
    conn.commit()
    conn.close()


init_db()

# ----------------------------
# Domain constants
# ----------------------------
JOB_TYPE_OPTIONS = [
    "Inspect & Collect",
    "Delivery",
    "Collection",
    "Inspection",
    "Other",
]

JOB_STATUS_OPTIONS = [
    "Open",
    "Completed",
    "Withdraw",
    "Aborted",
]

# Fee rules (adjust if your business rules differ)
INSPECT_COLLECT_FEE = 8.0  # the £8 you mentioned


# ----------------------------
# Normalization / Validation
# ----------------------------
def normalize_date(d: date) -> str:
    # Store in ISO: YYYY-MM-DD
    return d.isoformat()


def safe_upper(s: str) -> str:
    return (s or "").strip().upper()


def parse_amount(amount_raw: str) -> Tuple[bool, float, str]:
    """
    Amount is allowed empty => treat as 0.
    Otherwise must be a valid money-like number.
    """
    raw = (amount_raw or "").strip()
    if raw == "":
        return True, 0.0, ""
    if not re.fullmatch(r"\d+(\.\d{1,2})?", raw):
        return False, 0.0, "Amount must be a valid number (e.g., 8 or 8.20)."
    return True, float(raw), ""


def validate_entry(
    job_number: str,
    vehicle_description: str,
    amount_raw: str,
    waiting_time: str,
    job_type: str,
    job_status: str,
) -> Tuple[bool, str]:
    job_number = (job_number or "").strip()
    vehicle_description = (vehicle_description or "").strip()
    waiting_time = (waiting_time or "").strip()

    if job_type not in JOB_TYPE_OPTIONS:
        return False, "Job type is invalid."
    if job_status not in JOB_STATUS_OPTIONS:
        return False, "Job status is invalid."

    # Job number: digits only, reasonable length
    if not job_number:
        return False, "Job number is required."
    if not re.fullmatch(r"\d{6,12}", job_number):
        return False, "Job number must be 6–12 digits only."

    # Vehicle description: must contain letters; cannot be only numbers
    if not vehicle_description:
        return False, "Vehicle description is required."
    if re.fullmatch(r"\d+", vehicle_description):
        return False, "Vehicle description can’t be only numbers. Put the job number in Job number."
    if vehicle_description.strip() == job_number.strip():
        return False, "Vehicle description looks like a job number. Put it in Job number."
    if not re.search(r"[A-Za-z]", vehicle_description):
        return False, "Vehicle description must include letters (e.g., Make/Model)."

    ok_amt, _, amt_msg = parse_amount(amount_raw)
    if not ok_amt:
        return False, amt_msg

    # Waiting time: optional, but if present must match allowed patterns
    if waiting_time:
        ok_range = re.fullmatch(r"\d{1,2}\s*-\s*\d{1,2}", waiting_time)  # 10-12
        ok_clock = re.fullmatch(r"\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}", waiting_time)  # 10:30-12:15
        if not (ok_range or ok_clock):
            return False, "Waiting time must be like 10-12 or 10:30-12:15."

    return True, ""


# ----------------------------
# DB CRUD
# ----------------------------
def insert_row(
    work_date_iso: str,
    job_number: str,
    job_type: str,
    job_status: str,
    vehicle_description: str,
    amount: float,
    waiting_time: str,
    comment: str,
) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO worklog (
            created_at, work_date, job_number, job_type, job_status,
            vehicle_description, amount, waiting_time, comment
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(timespec="seconds"),
            work_date_iso,
            str(job_number).strip(),
            str(job_type).strip(),
            str(job_status).strip(),
            safe_upper(vehicle_description),
            float(amount),
            (waiting_time or "").strip(),
            (comment or "").strip(),
        ),
    )
    conn.commit()
    conn.close()


def update_row(row_id: int, fields: dict) -> None:
    """
    fields keys must match table columns.
    """
    allowed = {
        "work_date",
        "job_number",
        "job_type",
        "job_status",
        "vehicle_description",
        "amount",
        "waiting_time",
        "comment",
    }
    cleaned = {k: v for k, v in fields.items() if k in allowed}
    if "vehicle_description" in cleaned:
        cleaned["vehicle_description"] = safe_upper(str(cleaned["vehicle_description"]))
    if "job_number" in cleaned:
        cleaned["job_number"] = str(cleaned["job_number"]).strip()

    if not cleaned:
        return

    set_clause = ", ".join([f"{k} = ?" for k in cleaned.keys()])
    params = list(cleaned.values()) + [row_id]

    conn = get_conn()
    conn.execute(f"UPDATE worklog SET {set_clause} WHERE id = ?", params)
    conn.commit()
    conn.close()


def delete_row(row_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM worklog WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()


def load_df() -> pd.DataFrame:
    """
    Always fetch latest data from DB. No caching.
    """
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            id,
            created_at,
            work_date,
            job_number,
            job_type,
            job_status,
            vehicle_description,
            amount,
            waiting_time,
            comment
        FROM worklog
        ORDER BY work_date DESC, id DESC
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return df

    # normalize
    df["work_date"] = df["work_date"].astype(str)
    df["job_number"] = df["job_number"].astype(str).str.strip()
    df["job_type"] = df["job_type"].astype(str)
    df["job_status"] = df["job_status"].astype(str)
    df["vehicle_description"] = df["vehicle_description"].astype(str).str.upper()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    return df


# ----------------------------
# Calculations / Reporting
# ----------------------------
def compute_inspect_collect_fee(df: pd.DataFrame) -> pd.Series:
    """
    Fee applies only to Inspect & Collect jobs.
    IMPORTANT: Withdraw is excluded from the £8 total.
    """
    if df.empty:
        return pd.Series(dtype=float)

    jt = df["job_type"].fillna("").astype(str).str.strip()
    js = df["job_status"].fillna("").astype(str).str.strip().str.lower()

    fee = pd.Series(0.0, index=df.index)
    mask = (jt == "Inspect & Collect") & (js != "withdraw")
    fee.loc[mask] = INSPECT_COLLECT_FEE
    return fee


def compute_total_amount_of_work(df: pd.DataFrame) -> float:
    """
    Total amount of work:
    - sum(amount) EXCLUDING Withdraw
    """
    if df.empty:
        return 0.0
    js = df["job_status"].fillna("").astype(str).str.lower()
    return float(df.loc[js != "withdraw", "amount"].sum())


def compute_total_owned(df: pd.DataFrame) -> float:
    """
    "Total owned" (you previously said it was wrong).
    I'm defining it cleanly as:
      total_owned = (sum of amounts excluding Withdraw) + (inspect & collect fees excluding Withdraw)

    If your meaning is different, adjust here ONLY.
    """
    if df.empty:
        return 0.0
    total_work = compute_total_amount_of_work(df)
    ic_fee_total = float(compute_inspect_collect_fee(df).sum())
    return float(total_work + ic_fee_total)


# ----------------------------
# UI: Session helpers (clear fields except date)
# ----------------------------
ENTRY_KEYS = [
    "job_number",
    "job_type",
    "job_status",
    "vehicle_description",
    "amount",
    "waiting_time",
    "comment",
]


def clear_entry_fields_keep_date():
    # keep st.session_state["work_date"] intact
    for k in ENTRY_KEYS:
        # sensible reset defaults
        if k in ("job_type",):
            st.session_state[k] = "Inspect & Collect"
        elif k in ("job_status",):
            st.session_state[k] = "Open"
        else:
            st.session_state[k] = ""


# Initialize defaults
if "job_type" not in st.session_state:
    st.session_state["job_type"] = "Inspect & Collect"
if "job_status" not in st.session_state:
    st.session_state["job_status"] = "Open"
if "work_date" not in st.session_state:
    st.session_state["work_date"] = date.today()


# ----------------------------
# Tabs
# ----------------------------
tab_add, tab_view, tab_import, tab_dash, tab_admin = st.tabs(
    ["➕ Add Entry", "📄 View / Edit", "⬆️ Import / Export", "📊 Dashboard", "🛠️ Admin"]
)

# ----------------------------
# Add Entry
# ----------------------------
with tab_add:
    st.subheader("Add a new entry")

    with st.form("add_entry_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            work_date = st.date_input("Date", key="work_date")
            job_number = st.text_input("Job number (6–12 digits)", key="job_number")

        with c2:
            job_type = st.selectbox("Job type", JOB_TYPE_OPTIONS, key="job_type")
            job_status = st.selectbox("Job status", JOB_STATUS_OPTIONS, key="job_status")

        with c3:
            vehicle_description = st.text_input("Vehicle description", key="vehicle_description")
            amount_raw = st.text_input("Amount (£)", key="amount")
            waiting_time = st.text_input("Waiting time (optional)", key="waiting_time")

        comment = st.text_input("Comment (optional)", key="comment")

        submitted = st.form_submit_button("Save")

    if submitted:
        ok, msg = validate_entry(
            job_number=job_number,
            vehicle_description=vehicle_description,
            amount_raw=amount_raw,
            waiting_time=waiting_time,
            job_type=job_type,
            job_status=job_status,
        )
        if not ok:
            st.error(msg)
        else:
            ok_amt, amount, _ = parse_amount(amount_raw)
            insert_row(
                work_date_iso=normalize_date(work_date),
                job_number=job_number,
                job_type=job_type,
                job_status=job_status,
                vehicle_description=vehicle_description,  # stored uppercase inside insert_row
                amount=amount,
                waiting_time=waiting_time,
                comment=comment,
            )
            st.success("Saved ✅")
            clear_entry_fields_keep_date()
            st.rerun()

    st.divider()
    st.info(
        "Rules enforced: Job number must be digits only (6–12). "
        "Vehicle description must include letters and cannot be only numbers. "
        "Vehicle description is saved in UPPERCASE."
    )

# ----------------------------
# View / Edit
# ----------------------------
with tab_view:
    st.subheader("View / Edit entries")

    df = load_df()

    if df.empty:
        st.warning("No data yet.")
    else:
        # Filters
        f1, f2, f3, f4 = st.columns([1, 1, 1, 2])

        with f1:
            status_filter = st.multiselect(
                "Filter status",
                options=JOB_STATUS_OPTIONS,
                default=JOB_STATUS_OPTIONS,
            )
        with f2:
            type_filter = st.multiselect(
                "Filter job type",
                options=JOB_TYPE_OPTIONS,
                default=JOB_TYPE_OPTIONS,
            )
        with f3:
            date_from = st.text_input("Date from (YYYY-MM-DD)", value="")
            date_to = st.text_input("Date to (YYYY-MM-DD)", value="")
        with f4:
            q = st.text_input("Search (job number / vehicle / comment)", value="")

        fdf = df.copy()

        if status_filter:
            fdf = fdf[fdf["job_status"].isin(status_filter)]
        if type_filter:
            fdf = fdf[fdf["job_type"].isin(type_filter)]

        if date_from.strip():
            fdf = fdf[fdf["work_date"] >= date_from.strip()]
        if date_to.strip():
            fdf = fdf[fdf["work_date"] <= date_to.strip()]

        if q.strip():
            qs = q.strip().lower()
            fdf = fdf[
                fdf["job_number"].astype(str).str.lower().str.contains(qs, na=False)
                | fdf["vehicle_description"].astype(str).str.lower().str.contains(qs, na=False)
                | fdf["comment"].astype(str).str.lower().str.contains(qs, na=False)
            ]

        st.caption(f"Rows: {len(fdf)}")

        # Edit grid (simple + safe)
        # NOTE: Streamlit doesn't guarantee perfect inline edit constraints,
        # so we validate when saving per-row.
        edit_cols = [
            "id",
            "work_date",
            "job_number",
            "job_type",
            "job_status",
            "vehicle_description",
            "amount",
            "waiting_time",
            "comment",
        ]

        edited = st.data_editor(
            fdf[edit_cols],
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn(disabled=True),
                "work_date": st.column_config.TextColumn(help="YYYY-MM-DD"),
                "amount": st.column_config.NumberColumn(format="%.2f"),
            },
            key="data_editor_view",
        )

        # Save edited changes
        colA, colB = st.columns([1, 3])
        with colA:
            if st.button("Save edits"):
                # Compare edited to original and write diffs
                orig = fdf[edit_cols].copy()
                new = edited.copy()

                changes = 0
                errors = 0

                # align by id
                orig_by_id = orig.set_index("id")
                new_by_id = new.set_index("id")

                for row_id in new_by_id.index:
                    if row_id not in orig_by_id.index:
                        continue
                    before = orig_by_id.loc[row_id]
                    after = new_by_id.loc[row_id]

                    diff_fields = {}
                    for col in edit_cols:
                        if col == "id":
                            continue
                        if pd.isna(before[col]) and pd.isna(after[col]):
                            continue
                        if str(before[col]) != str(after[col]):
                            diff_fields[col] = after[col]

                    if diff_fields:
                        # Validate if core fields changed or always validate
                        ok, msg = validate_entry(
                            job_number=str(after["job_number"]),
                            vehicle_description=str(after["vehicle_description"]),
                            amount_raw=str(after["amount"]) if after["amount"] is not None else "",
                            waiting_time=str(after["waiting_time"]) if after["waiting_time"] is not None else "",
                            job_type=str(after["job_type"]),
                            job_status=str(after["job_status"]),
                        )
                        if not ok:
                            errors += 1
                            st.error(f"Row id {row_id}: {msg}")
                            continue

                        # Normalize
                        if "work_date" in diff_fields:
                            diff_fields["work_date"] = str(diff_fields["work_date"]).strip()
                        if "amount" in diff_fields:
                            try:
                                diff_fields["amount"] = float(diff_fields["amount"])
                            except Exception:
                                st.error(f"Row id {row_id}: amount invalid")
                                errors += 1
                                continue

                        update_row(int(row_id), diff_fields)
                        changes += 1

                if errors == 0:
                    st.success(f"Saved {changes} row(s).")
                    st.rerun()
                else:
                    st.warning(f"Saved {changes} row(s). {errors} row(s) blocked by validation.")

        with colB:
            st.write("Tip: If you edit job_number / vehicle_description wrongly, Save will block it.")

# ----------------------------
# Import / Export
# ----------------------------
with tab_import:
    st.subheader("Import / Export")

    st.markdown(
        """
**Import rules**
- Expected columns (case-insensitive): `Date`, `Job number`, `Job type`, `Job status`, `Vehicle description`, `Amount`, `Waiting time`, `Comment`
- Vehicle description will be forced to UPPERCASE
- Bad rows will be rejected (same validation as manual entry)
"""
    )

    up = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "csv"])
    if up is not None:
        try:
            if up.name.lower().endswith(".xlsx"):
                data = pd.read_excel(up)
            else:
                data = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            data = None

        if data is not None:
            # Normalize columns
            cols = {c.strip().lower(): c for c in data.columns}
            def pick(*names):
                for n in names:
                    if n in cols:
                        return cols[n]
                return None

            c_date = pick("date", "work_date", "work date")
            c_job = pick("job number", "job_number", "job")
            c_type = pick("job type", "job_type", "type")
            c_status = pick("job status", "job_status", "status")
            c_vehicle = pick("vehicle description", "vehicle", "description")
            c_amount = pick("amount", "amt")
            c_wait = pick("waiting time", "waiting_time", "waiting")
            c_comment = pick("comment", "notes", "note")

            needed = [c_date, c_job, c_type, c_status, c_vehicle]
            if any(x is None for x in needed):
                st.error("Missing required columns. Need: Date, Job number, Job type, Job status, Vehicle description.")
            else:
                preview = data.head(20)
                st.dataframe(preview, use_container_width=True)

                if st.button("Import now"):
                    ok_rows = 0
                    bad_rows = 0

                    for i, r in data.iterrows():
                        # Date normalize
                        raw_date = r[c_date]
                        try:
                            if isinstance(raw_date, (datetime, pd.Timestamp)):
                                d_iso = raw_date.date().isoformat()
                            elif isinstance(raw_date, date):
                                d_iso = raw_date.isoformat()
                            else:
                                d_iso = str(raw_date).strip()
                                # basic format check
                                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d_iso):
                                    # try parse
                                    d_iso = pd.to_datetime(d_iso).date().isoformat()
                        except Exception:
                            bad_rows += 1
                            continue

                        job_number = str(r[c_job]).strip()
                        job_type = str(r[c_type]).strip()
                        job_status = str(r[c_status]).strip()
                        vehicle = str(r[c_vehicle]).strip()

                        amount_raw = ""
                        if c_amount is not None and not pd.isna(r[c_amount]):
                            amount_raw = str(r[c_amount]).strip()

                        waiting = ""
                        if c_wait is not None and not pd.isna(r[c_wait]):
                            waiting = str(r[c_wait]).strip()

                        comment = ""
                        if c_comment is not None and not pd.isna(r[c_comment]):
                            comment = str(r[c_comment]).strip()

                        # Map/clean type/status to known values if possible
                        if job_type not in JOB_TYPE_OPTIONS:
                            job_type = "Other"
                        if job_status not in JOB_STATUS_OPTIONS:
                            job_status = "Open"

                        ok, msg = validate_entry(job_number, vehicle, amount_raw, waiting, job_type, job_status)
                        if not ok:
                            bad_rows += 1
                            continue

                        ok_amt, amt, _ = parse_amount(amount_raw)
                        if not ok_amt:
                            bad_rows += 1
                            continue

                        insert_row(
                            work_date_iso=d_iso,
                            job_number=job_number,
                            job_type=job_type,
                            job_status=job_status,
                            vehicle_description=vehicle,
                            amount=amt,
                            waiting_time=waiting,
                            comment=comment,
                        )
                        ok_rows += 1

                    st.success(f"Imported {ok_rows} row(s). Rejected {bad_rows} bad row(s).")
                    st.rerun()

    st.divider()
    st.subheader("Export")
    df = load_df()
    if df.empty:
        st.info("Nothing to export yet.")
    else:
        out = df.copy()
        # Add computed fee column for convenience
        out["inspect_collect_fee"] = compute_inspect_collect_fee(out)
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="worklog_export.csv", mime="text/csv")

        # Excel export
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="worklog")
        st.download_button(
            "Download Excel",
            data=bio.getvalue(),
            file_name="worklog_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ----------------------------
# Dashboard
# ----------------------------
with tab_dash:
    st.subheader("Dashboard")

    df = load_df()

    if df.empty:
        st.warning("No data to display.")
    else:
        # Date filter
        d1, d2, d3 = st.columns([1, 1, 2])
        with d1:
            dash_from = st.text_input("From (YYYY-MM-DD)", value="")
        with d2:
            dash_to = st.text_input("To (YYYY-MM-DD)", value="")
        with d3:
            show_statuses = st.multiselect(
                "Show job statuses in the table",
                options=JOB_STATUS_OPTIONS,
                default=JOB_STATUS_OPTIONS,  # includes Withdraw/Aborted as requested
            )

        ddf = df.copy()
        if dash_from.strip():
            ddf = ddf[ddf["work_date"] >= dash_from.strip()]
        if dash_to.strip():
            ddf = ddf[ddf["work_date"] <= dash_to.strip()]
        if show_statuses:
            ddf = ddf[ddf["job_status"].isin(show_statuses)]

        ic_fee = compute_inspect_collect_fee(ddf)
        total_ic_fee = float(ic_fee.sum())

        total_work = compute_total_amount_of_work(ddf)  # excludes Withdraw
        total_owned = compute_total_owned(ddf)          # excludes Withdraw via function + adds IC fee

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total amount of work (excl. Withdraw)", f"£{total_work:,.2f}")
        k2.metric("Inspect & Collect £8 total (excl. Withdraw)", f"£{total_ic_fee:,.2f}")
        k3.metric("Total owned", f"£{total_owned:,.2f}")
        k4.metric("Rows (filtered)", f"{len(ddf)}")

        st.divider()

        # Table view
        show = ddf.copy()
        show["inspect_collect_fee"] = ic_fee
        cols = [
            "work_date",
            "job_number",
            "job_type",
            "job_status",
            "vehicle_description",
            "amount",
            "inspect_collect_fee",
            "waiting_time",
            "comment",
        ]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)

        st.divider()

        # Quick breakdowns
        b1, b2 = st.columns(2)
        with b1:
            st.caption("By job status (counts)")
            st.dataframe(ddf["job_status"].value_counts().rename("count").to_frame(), use_container_width=True)
        with b2:
            st.caption("By job type (sum amount, excl. Withdraw)")
            tmp = ddf.copy()
            tmp = tmp[tmp["job_status"].str.lower() != "withdraw"]
            sums = tmp.groupby("job_type")["amount"].sum().sort_values(ascending=False)
            st.dataframe(sums.rename("amount_sum").to_frame(), use_container_width=True)

# ----------------------------
# Admin
# ----------------------------
with tab_admin:
    st.subheader("Admin tools")

    df = load_df()
    if df.empty:
        st.info("No data.")
    else:
        st.markdown("### Delete a single row by ID")
        st.write("Use this if you have duplicates and want to delete just one entry.")
        del_id = st.number_input("Row ID to delete", min_value=1, step=1, value=1)
        if st.button("Delete row"):
            delete_row(int(del_id))
            st.success(f"Deleted row id {int(del_id)}")
            st.rerun()

        st.divider()
        st.markdown("### Find duplicates by Job number + Date")
        dup = (
            df.groupby(["work_date", "job_number"])
            .size()
            .reset_index(name="count")
            .query("count > 1")
            .sort_values(["count", "work_date"], ascending=[False, False])
        )
        if dup.empty:
            st.success("No duplicates found by (date + job number).")
        else:
            st.dataframe(dup, use_container_width=True, hide_index=True)
            st.caption("Use View/Edit tab to locate the rows, then delete by ID above.")

st.caption("If something looks off after deploy, the version at the top tells you which code is running.")