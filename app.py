import sqlite3
import subprocess
from datetime import date
import pandas as pd
import streamlit as st

from worklog.db import (
    ensure_schema,
    read_all,
    get_db_path,
    backfill_from_dataframe,
    update_status_for_year,
)
from worklog.normalize import (
    STATUS_OPTIONS,
    JOB_TYPE_OPTIONS,
    JOB_EXPENSE_OPTIONS,
    NON_PAYABLE_STATUSES,
    normalize_status,
    normalize_job_type,
    normalize_expense_type,
    clean_job_number,
    zero_to_none,
)
from worklog.dates import to_clean_date_series, safe_date_bounds, week_start
from worklog.waiting import parse_waiting_time, WAITING_RATE
from worklog.dedupe import dedup_for_reporting

APP_TITLE = "Worklog"

INSPECT_COLLECT_RATE = 8.00
INSPECT_COLLECT_TYPES = {"Inspect and Collect", "Inspect and Collect 2"}

UI_COLUMNS = [
    "Date",
    "job number",
    "job type",
    "vehcile description",
    "vehicle Reg",
    "collection from",
    "delivery to",
    "job amount",
    "Job Expenses",
    "expenses Amount",
    "Auth code",
    "job status",
    "waiting time",
    "comments",
]


def get_live_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def filter_money_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    s = df["job_status"].fillna("").astype(str).str.lower().str.strip()
    return df[~s.isin(NON_PAYABLE_STATUSES)].copy()


def get_conn():
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def insert_row(
    work_date_val: date,
    job_number: str,
    job_type: str,
    vehicle_description: str,
    vehicle_reg: str,
    collection_from: str,
    delivery_to: str,
    job_amount,
    job_expenses: str,
    expenses_amount,
    auth_code: str,
    job_status: str,
    waiting_time_raw: str,
    comments: str,
):
    from worklog.db import TABLE_NAME

    job_number = clean_job_number(job_number)
    wd = work_date_val.isoformat()

    w_hours, w_norm = parse_waiting_time(waiting_time_raw)
    w_amount = (w_hours * WAITING_RATE) if w_hours is not None else None

    status = normalize_status(job_status)
    jt = normalize_job_type(job_type)
    je = normalize_expense_type(job_expenses)

    vdesc = str(vehicle_description or "").strip().upper()
    vreg = str(vehicle_reg or "").strip()
    cfrom = str(collection_from or "").strip()
    cto = str(delivery_to or "").strip()
    auth = str(auth_code or "").strip()
    cmts = str(comments or "").strip()

    job_amount_db = zero_to_none(job_amount)
    expenses_amount_db = zero_to_none(expenses_amount)

    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME}
              (
                work_date, job_id, category,
                vehicle_description, vehicle_reg, collection_from, delivery_to,
                amount, job_expenses, expenses_amount, auth_code, job_status,
                waiting_time, waiting_hours, waiting_amount,
                description, hours, comments
              )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                wd,
                job_number,
                jt,
                vdesc,
                vreg,
                cfrom,
                cto,
                job_amount_db,
                je,
                expenses_amount_db,
                auth,
                status,
                w_norm or "",
                w_hours,
                w_amount,
                "",
                None,
                cmts,
            ),
        )
        conn.commit()


def update_row_by_id(row_id: int, **fields):
    from worklog.db import TABLE_NAME

    with get_conn() as conn:
        cols = []
        vals = []
        for k, v in fields.items():
            cols.append(f"{k} = ?")
            vals.append(v)
        vals.append(int(row_id))
        conn.execute(f"UPDATE {TABLE_NAME} SET {', '.join(cols)} WHERE id = ?", vals)
        conn.commit()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

ensure_schema()

with st.sidebar:
    st.header("Actions")
    st.caption("DB in use:")
    st.code(get_db_path())
    st.caption(f"Live version: {get_live_version()}")

    st.divider()
    st.subheader("Backfill missing fields (NO duplicates)")
    st.caption(
        "Upload your Excel again and this will FILL missing vehicle fields for rows already in the DB (match by Date + Job)."
    )
    up_backfill = st.file_uploader("Backfill file", type=["xlsx", "xls", "csv"], key="backfill_file")
    if up_backfill is not None:
        try:
            bf_df = (
                pd.read_csv(up_backfill)
                if up_backfill.name.lower().endswith(".csv")
                else pd.read_excel(up_backfill)
            )
            st.dataframe(bf_df.head(15), use_container_width=True)
            if st.button("Run backfill now"):
                matched, updated = backfill_from_dataframe(bf_df)
                st.success(f"Backfill done. Matched: {matched}, Updated: {updated}")
                st.rerun()
        except Exception as e:
            st.error(f"Backfill failed: {e}")

    st.divider()
    st.subheader("Admin: set 2025 jobs to Paid")
    if st.button("Set all 2025 rows to Paid"):
        changed = update_status_for_year(2025, "Paid")
        st.success(f"Updated {changed} rows to Paid.")
        st.rerun()

df_all = read_all()
if df_all.empty:
    st.info("Database is empty.")
    st.stop()

# normalize display
df_all["work_date"] = to_clean_date_series(df_all["work_date"])
df_all["job_id"] = df_all["job_id"].apply(clean_job_number)
df_all["category"] = df_all["category"].fillna("").astype(str).apply(normalize_job_type)
df_all["job_status"] = df_all["job_status"].apply(normalize_status)

df_all["vehicle_description"] = (
    df_all["vehicle_description"].fillna("").astype(str).str.strip().str.upper()
)
df_all["vehicle_reg"] = df_all["vehicle_reg"].fillna("").astype(str).str.strip()
df_all["collection_from"] = df_all["collection_from"].fillna("").astype(str).str.strip()
df_all["delivery_to"] = df_all["delivery_to"].fillna("").astype(str).str.strip()
df_all["job_expenses"] = df_all["job_expenses"].fillna("").astype(str).apply(normalize_expense_type)
df_all["auth_code"] = df_all["auth_code"].fillna("").astype(str).str.strip()
df_all["comments"] = df_all["comments"].fillna("").astype(str).str.strip()

min_d, max_d = safe_date_bounds(df_all["work_date"])
default_end = max(max_d, date.today())

with st.sidebar:
    st.header("Filters")
    date_val = st.date_input(
        "Work date range", value=(min_d, default_end), min_value=min_d, max_value=default_end
    )
    start_d, end_d = (date_val if isinstance(date_val, (tuple, list)) else (date_val, date_val))
    status_filter = st.multiselect("Job status", options=STATUS_OPTIONS, default=STATUS_OPTIONS)
    search_txt = st.text_input("Search (job / reg / auth / locations / comments)", value="").strip()

df = df_all.copy()
df = df[df["work_date"].notna()]
df = df[(df["work_date"] >= start_d) & (df["work_date"] <= end_d)]
df = df[df["job_status"].isin(status_filter)]
if search_txt:
    m = (
        df["job_id"].astype(str).str.contains(search_txt, case=False, na=False)
        | df["vehicle_reg"].astype(str).str.contains(search_txt, case=False, na=False)
        | df["auth_code"].astype(str).str.contains(search_txt, case=False, na=False)
        | df["collection_from"].astype(str).str.contains(search_txt, case=False, na=False)
        | df["delivery_to"].astype(str).str.contains(search_txt, case=False, na=False)
        | df["comments"].astype(str).str.contains(search_txt, case=False, na=False)
    )
    df = df[m]

tab1, tab2 = st.tabs(["Add entry", "View & Report"])

with tab1:
    st.subheader("Add entry")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        work_date_val = st.date_input("Date", value=date.today())
        job_number = st.text_input("job number (required)")
        job_type = st.selectbox("job type", JOB_TYPE_OPTIONS, index=0)
        job_status = st.selectbox(
            "job status", STATUS_OPTIONS, index=STATUS_OPTIONS.index("Pending")
        )

    with c2:
        vehicle_description = st.text_input("vehcile description")
        vehicle_reg = st.text_input("vehicle Reg")

    with c3:
        collection_from = st.text_input("collection from")
        delivery_to = st.text_input("delivery to")

    with c4:
        job_amount = st.number_input("job amount", step=0.5, value=0.0)
        job_expenses = st.selectbox("Job Expenses", JOB_EXPENSE_OPTIONS, index=0)
        expenses_amount = st.number_input("expenses Amount", step=0.5, value=0.0)
        auth_code = st.text_input("Auth code")
        waiting_time_raw = st.text_input(
            "waiting time (e.g. 10-12 or 10:30-12:15)", value=""
        )

        w_hours, w_norm = parse_waiting_time(waiting_time_raw)
        if waiting_time_raw.strip():
            if w_hours is None:
                st.error("Waiting time format invalid.")
            else:
                st.write(
                    f"Waiting: **{w_norm}** | Hours: **{w_hours:.2f}** | Owed: **£{(w_hours*WAITING_RATE):.2f}**"
                )

    comments = st.text_area("comments")

    if st.button("Save entry"):
        jn = clean_job_number(job_number)
        if not jn:
            st.error("job number is required.")
        elif waiting_time_raw.strip() and w_hours is None:
            st.error("Fix waiting time format before saving.")
        else:
            try:
                insert_row(
                    work_date_val=work_date_val,
                    job_number=jn,
                    job_type=job_type,
                    vehicle_description=vehicle_description,
                    vehicle_reg=vehicle_reg,
                    collection_from=collection_from,
                    delivery_to=delivery_to,
                    job_amount=job_amount,
                    job_expenses=job_expenses,
                    expenses_amount=expenses_amount,
                    auth_code=auth_code,
                    job_status=job_status,
                    waiting_time_raw=waiting_time_raw,
                    comments=comments,
                )
                st.success("Saved.")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("Duplicate blocked: this Date + Job number already exists.")
            except Exception as e:
                st.error(f"Save failed: {e}")

with tab2:
    st.subheader("View & Report")

    if df.empty:
        st.info("No records in this range.")
        st.stop()

    st.markdown("### Dashboard (includes Withdraw + Aborted)")
    status_counts = df["job_status"].value_counts()
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Withdraw jobs", int(status_counts.get("Withdraw", 0)))
    d2.metric("Aborted jobs", int(status_counts.get("Aborted", 0)))
    d3.metric("Completed jobs", int(status_counts.get("Completed", 0)))
    d4.metric("Paid jobs", int(status_counts.get("Paid", 0)))

    report_df = dedup_for_reporting(df)
    money_df = filter_money_rows(report_df)

    total_job = pd.to_numeric(money_df["amount"], errors="coerce").fillna(0).sum()
    total_exp = pd.to_numeric(money_df["expenses_amount"], errors="coerce").fillna(0).sum()
    total_wait = pd.to_numeric(money_df["waiting_amount"], errors="coerce").fillna(0).sum()
    total_owed = total_job + total_exp + total_wait

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Job amount", f"£{total_job:,.2f}")
    k2.metric("Expenses", f"£{total_exp:,.2f}")
    k3.metric("Waiting owed", f"£{total_wait:,.2f}")
    k4.metric("Total owed", f"£{total_owed:,.2f}")

    st.divider()
    st.markdown("### Inspect & Collect (£8/job)")
    ic = report_df[report_df["category"].isin(INSPECT_COLLECT_TYPES)].copy()
    ic = filter_money_rows(ic)
    ic_count = len(ic)
    ic_total = ic_count * INSPECT_COLLECT_RATE
    a1, a2 = st.columns(2)
    a1.metric("Count", f"{ic_count:,}")
    a2.metric("Total owed", f"£{ic_total:,.2f}")

    st.divider()
    st.markdown("### Records (strict column order)")
    view_df = report_df.copy().rename(
        columns={
            "work_date": "Date",
            "job_id": "job number",
            "category": "job type",
            "vehicle_description": "vehcile description",
            "vehicle_reg": "vehicle Reg",
            "collection_from": "collection from",
            "delivery_to": "delivery to",
            "amount": "job amount",
            "job_expenses": "Job Expenses",
            "expenses_amount": "expenses Amount",
            "auth_code": "Auth code",
            "job_status": "job status",
            "waiting_time": "waiting time",
            "comments": "comments",
        }
    )
    for col in UI_COLUMNS:
        if col not in view_df.columns:
            view_df[col] = ""

    st.dataframe(view_df[UI_COLUMNS], use_container_width=True, hide_index=True)

    # =========================
    # Weekly summary (FIXED)
    # =========================
    st.divider()
    st.subheader("Weekly summary (deduped by Date+Job)")

    dfw = filter_money_rows(report_df).copy()
    dfw["week_start"] = dfw["work_date"].apply(week_start)

    weekly = (
        dfw.groupby("week_start", as_index=False)
        .agg(
            rows=("id", "count"),  # if "id" ever doesn't exist, change to ("job_id", "count")
            job_amount=("amount", "sum"),
            expenses_amount=("expenses_amount", "sum"),
            waiting_owed=("waiting_amount", "sum"),
        )
        .sort_values("week_start", ascending=False)
    )

    for c in ["job_amount", "expenses_amount", "waiting_owed"]:
        weekly[c] = pd.to_numeric(weekly[c], errors="coerce").fillna(0)

    weekly["total_owed"] = weekly["job_amount"] + weekly["waiting_owed"] + weekly["expenses_amount"]

    st.dataframe(weekly, use_container_width=True, hide_index=True)