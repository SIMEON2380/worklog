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

<<<<<<< HEAD
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
=======
# Inspect & Collect pay rate
INSPECT_COLLECT_RATE = 8.00
INSPECT_COLLECT_TYPES = {"Inspect and Collect", "Inspect and Collect 2"}

# NOTE: kept your existing labels (including typos) to avoid breaking UI expectations
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

EXPECTED_DB_COLS = [
    "id",
    "work_date",
    "description",
    "hours",
    "amount",
    "job_id",
    "category",
    "job_status",
    "waiting_time",
    "waiting_hours",
    "waiting_amount",
    "vehicle_description",
    "vehicle_reg",
    "collection_from",
    "delivery_to",
    "job_expenses",
    "expenses_amount",
    "auth_code",
    "comments",
    "created_at",
]
>>>>>>> 30ef248 (Fix weekly summary column KeyError)

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

<<<<<<< HEAD
    df = list_jobs(search=search, limit=int(limit))
=======
def clean_job_number(val: Any) -> str:
    """
    Fix Excel numeric job ids like 11623733.0 -> 11623733
    Keeps text values as-is.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return ""
    s = s.replace(",", "")
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s.strip()


def ensure_db_dir():
    os.makedirs(DB_DIR, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    ensure_db_dir()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def table_columns(conn: sqlite3.Connection, table_name: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows}


def ensure_schema():
    with get_conn() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                work_date TEXT,
                description TEXT,
                hours REAL,
                amount REAL,
                job_id TEXT,
                category TEXT,
                job_status TEXT,

                waiting_time TEXT,
                waiting_hours REAL,
                waiting_amount REAL,

                vehicle_description TEXT,
                vehicle_reg TEXT,
                collection_from TEXT,
                delivery_to TEXT,
                job_expenses TEXT,
                expenses_amount REAL,
                auth_code TEXT,

                comments TEXT,

                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()

        cols = table_columns(conn, TABLE_NAME)
        migrations = [
            ("work_date", "TEXT"),
            ("description", "TEXT"),
            ("hours", "REAL"),
            ("amount", "REAL"),
            ("job_id", "TEXT"),
            ("category", "TEXT"),
            ("job_status", "TEXT"),
            ("waiting_time", "TEXT"),
            ("waiting_hours", "REAL"),
            ("waiting_amount", "REAL"),
            ("vehicle_description", "TEXT"),
            ("vehicle_reg", "TEXT"),
            ("collection_from", "TEXT"),
            ("delivery_to", "TEXT"),
            ("job_expenses", "TEXT"),
            ("expenses_amount", "REAL"),
            ("auth_code", "TEXT"),
            ("comments", "TEXT"),
            ("created_at", "TEXT"),
        ]
        for col, col_type in migrations:
            if col not in cols:
                conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {col} {col_type}")
        conn.commit()

        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_work_date ON {TABLE_NAME}(work_date)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_job_id ON {TABLE_NAME}(job_id)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_status ON {TABLE_NAME}(job_status)")
        conn.commit()


# =========================
# Date parsing
# =========================
def to_clean_date_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="object")

    s2 = s.copy()

    if pd.api.types.is_datetime64_any_dtype(s2):
        return s2.dt.date

    s_str = s2.astype("string").str.strip()
    out = pd.Series([pd.NaT] * len(s2), index=s2.index, dtype="object")

    iso = pd.to_datetime(s_str, format="%Y-%m-%d", errors="coerce")
    iso_mask = iso.notna()
    if iso_mask.any():
        out.loc[iso_mask] = iso[iso_mask].dt.date

    rem = ~iso_mask
    num = pd.to_numeric(s_str.where(rem, pd.NA), errors="coerce")
    num_mask = num.notna()
    if num_mask.any():
        n = num[num_mask]

        ymd_mask = (n >= 19000101) & (n <= 21001231)
        if ymd_mask.any():
            ymd_vals = n[ymd_mask].astype("int64").astype(str)
            ymd_dt = pd.to_datetime(ymd_vals, format="%Y%m%d", errors="coerce")
            out.loc[ymd_dt.index] = ymd_dt.dt.date

        rest = n[~ymd_mask]

        ms_mask = rest >= 10_000_000_000
        if ms_mask.any():
            ms_dt = pd.to_datetime(rest[ms_mask], unit="ms", errors="coerce", utc=False)
            out.loc[ms_dt.index] = ms_dt.dt.date

        sec_rest = rest[~ms_mask]
        sec_mask = (sec_rest >= 1_000_000_000) & (sec_rest <= 4_000_000_000)
        if sec_mask.any():
            sec_dt = pd.to_datetime(sec_rest[sec_mask], unit="s", errors="coerce", utc=False)
            out.loc[sec_dt.index] = sec_dt.dt.date

        excel_rest = sec_rest[~sec_mask]
        excel_mask = (excel_rest >= 20000) & (excel_rest <= 80000)
        if excel_mask.any():
            ex = excel_rest[excel_mask]
            ex_dt = pd.to_datetime(ex, unit="D", origin="1899-12-30", errors="coerce")
            out.loc[ex_dt.index] = ex_dt.dt.date

        still_nat_idx = out[out.isna()].index.intersection(excel_rest.index)
        if len(still_nat_idx) > 0:
            ex2 = excel_rest.loc[still_nat_idx]
            ex2_dt = pd.to_datetime(ex2, unit="D", origin="1904-01-01", errors="coerce")
            out.loc[ex2_dt.index] = ex2_dt.dt.date

    rem2 = out.isna()
    if rem2.any():
        dt_rest = pd.to_datetime(s_str[rem2], errors="coerce", dayfirst=True)
        out.loc[dt_rest.index] = dt_rest.dt.date

    # sanity clamp
    min_ok = date(2000, 1, 1)
    max_ok = date(2100, 12, 31)

    def clamp(d):
        if pd.isna(d) or d is None:
            return pd.NaT
        if isinstance(d, date) and (d < min_ok or d > max_ok):
            return pd.NaT
        return d

    return out.apply(clamp)


def safe_date_bounds(dates: Optional[pd.Series]) -> Tuple[date, date]:
    today = date.today()
    if dates is None or len(dates) == 0:
        return today, today
    s = pd.Series(dates)
    s = to_clean_date_series(s).dropna()
    if s.empty:
        return today, today
    vals = s.tolist()
    return min(vals), max(vals)


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


# =========================
# Waiting parsing
# =========================
def parse_waiting_time(text: str) -> Tuple[Optional[float], Optional[str]]:
    if text is None:
        return None, None
    raw = str(text).strip()
    if raw == "":
        return None, None

    raw = raw.replace("–", "-").replace("—", "-")
    raw = re.sub(r"\s+", "", raw)
    if raw.count("-") != 1:
        return None, None

    start_s, end_s = raw.split("-")

    def to_minutes(t: str) -> Optional[int]:
        if t == "":
            return None
        if re.match(r"^\d{1,2}\.\d{2}$", t):
            t = t.replace(".", ":")
        if ":" in t:
            parts = t.split(":")
            if len(parts) != 2:
                return None
            try:
                h = int(parts[0])
                m = int(parts[1])
            except ValueError:
                return None
            if not (0 <= h <= 47 and 0 <= m <= 59):
                return None
            return h * 60 + m
        if re.match(r"^\d+(\.\d+)?$", t):
            try:
                val = float(t)
            except ValueError:
                return None
            h = int(val)
            frac = val - h
            m = int(round(frac * 60))
            if m == 60:
                h += 1
                m = 0
            if not (0 <= h <= 47 and 0 <= m <= 59):
                return None
            return h * 60 + m
        return None

    start_min = to_minutes(start_s)
    end_min = to_minutes(end_s)
    if start_min is None or end_min is None:
        return None, None

    diff = end_min - start_min
    if diff < 0:
        diff += 24 * 60

    hours = diff / 60.0

    def fmt(mins: int) -> str:
        mins = mins % (24 * 60)
        h = mins // 60
        m = mins % 60
        return f"{h:02d}:{m:02d}"

    return hours, f"{fmt(start_min)}-{fmt(end_min)}"


# =========================
# Data access
# =========================
def read_all() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY work_date DESC, id DESC", conn)

>>>>>>> 30ef248 (Fix weekly summary column KeyError)
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

<<<<<<< HEAD
            if row is None:
                st.warning("Job number not found.")
=======
# -------- Upload --------
with tab2:
    st.subheader("Upload Excel/CSV")
    st.write(
        "Required: **Date** (or `work_date`) and **job number** (or `job_id`/`job_number`). "
        "Optional: job type, vehicle fields, amounts, auth code, status, waiting time, comments."
    )

    up = st.file_uploader("Choose a file", type=["xlsx", "xls", "csv"])
    if up is not None:
        try:
            up_df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
            st.dataframe(up_df.head(30), use_container_width=True)
            if st.button("Import into database"):
                n = insert_many(up_df)
                st.success(f"Imported {n} rows.")
                st.rerun()
        except Exception as e:
            st.error(f"Upload/import failed: {e}")

# -------- View & Edit --------
with tab3:
    st.subheader("View & Edit")

    if df.empty:
        st.info("No records in this range.")
    else:
        # Dashboard counts should include Withdraw/Aborted
        st.markdown("### Dashboard (includes Withdraw + Aborted)")
        status_counts = df["job_status"].fillna("Unknown").value_counts()
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Withdraw jobs", int(status_counts.get("Withdraw", 0)))
        d2.metric("Aborted jobs", int(status_counts.get("Aborted", 0)))
        d3.metric("Completed jobs", int(status_counts.get("Completed", 0)))
        d4.metric("Paid jobs", int(status_counts.get("Paid", 0)))

        # IMPORTANT: use deduped rows for totals
        report_df = dedup_for_reporting(df)

        # Withdraw should NOT be included in money totals
        money_df = report_df[report_df["job_status"].astype(str).str.lower() != "withdraw"].copy()

        total_job = pd.to_numeric(money_df["amount"], errors="coerce").fillna(0).sum()
        total_exp = pd.to_numeric(money_df["expenses_amount"], errors="coerce").fillna(0).sum()
        total_wait = pd.to_numeric(money_df["waiting_amount"], errors="coerce").fillna(0).sum()

        total_earned = total_job + total_exp + total_wait

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Rows (filtered)", f"{len(df):,}")
        k2.metric("Job amount (deduped, no Withdraw)", f"£{total_job:,.2f}")
        k3.metric("Expenses (deduped, no Withdraw)", f"£{total_exp:,.2f}")
        k4.metric("Waiting owed (deduped, no Withdraw)", f"£{total_wait:,.2f}")
        k5.metric("Total owed (fixed)", f"£{total_earned:,.2f}")

        # =========================
        # Inspect & Collect table (£8/job)
        # =========================
        st.divider()
        st.markdown("### Inspect & Collect (£8 per job)")

        ic = report_df[report_df["category"].isin(INSPECT_COLLECT_TYPES)].copy()

        ic_count = int(len(ic))
        ic_total = ic_count * INSPECT_COLLECT_RATE

        x1, x2 = st.columns(2)
        x1.metric("Inspect & Collect jobs (deduped)", f"{ic_count:,}")
        x2.metric("Total owed (£8/job)", f"£{ic_total:,.2f}")

        if ic.empty:
            st.caption("No Inspect & Collect jobs in the current filters.")
        else:
            ic_view = ic[["work_date", "job_id", "vehicle_reg", "category", "job_status"]].copy()
            ic_view = ic_view.rename(
                columns={
                    "work_date": "Date",
                    "job_id": "job number",
                    "vehicle_reg": "vehicle Reg",
                    "category": "job type",
                    "job_status": "job status",
                }
            )
            ic_view["Pay (£)"] = INSPECT_COLLECT_RATE

            total_row = {
                "Date": "",
                "job number": "",
                "vehicle Reg": "",
                "job type": "TOTAL",
                "job status": f"{ic_count} job(s)",
                "Pay (£)": ic_total,
            }
            ic_view = pd.concat([ic_view, pd.DataFrame([total_row])], ignore_index=True)

            st.dataframe(ic_view, use_container_width=True, hide_index=True)

        # Optional view: list Withdraw/Aborted rows in this filtered range
        st.divider()
        st.markdown("### Withdraw / Aborted jobs in this view")
        wa = df[df["job_status"].isin(["Withdraw", "Aborted"])].copy()
        if wa.empty:
            st.caption("None in the current filters.")
        else:
            wa_view = wa.rename(
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
                if col not in wa_view.columns:
                    wa_view[col] = ""
            st.dataframe(wa_view[UI_COLUMNS], use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("## Edit (select a job number and save)")

        all_jobs = sorted([jn for jn in df_all["job_id"].dropna().astype(str).unique().tolist() if jn])
        if not all_jobs:
            st.info("No job numbers found.")
        else:
            if st.session_state.edit_selected_job not in all_jobs:
                st.session_state.edit_selected_job = all_jobs[0]

            selected_job_number = st.selectbox(
                "Select job number",
                options=all_jobs,
                index=all_jobs.index(st.session_state.edit_selected_job),
                key="edit_job_select",
            )
            st.session_state.edit_selected_job = selected_job_number

            matches = read_rows_by_job_number(selected_job_number)
            if matches.empty:
                st.error("Could not load that job number from the database.")
>>>>>>> 30ef248 (Fix weekly summary column KeyError)
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

<<<<<<< HEAD
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
=======
                    new_comments = st.text_area("comments", value=cur_comments, key=f"e_comments_{row_id}_{nonce}")

                    save = st.form_submit_button("Save changes")
                    if save:
                        jn = clean_job_number(new_job_number)
                        if jn == "":
                            st.error("job number is required.")
                        elif new_waiting_raw.strip() and wh is None:
                            st.error("Fix waiting time format before saving.")
                        else:
                            update_row_by_id(
                                row_id=row_id,
                                work_date_val=new_date,
                                job_number=jn,
                                job_type=new_job_type,
                                vehicle_description=new_vdesc,
                                vehicle_reg=new_vreg,
                                collection_from=new_from,
                                delivery_to=new_to,
                                job_amount=float(new_job_amount),
                                job_expenses=new_job_expenses,
                                expenses_amount=float(new_exp_amt),
                                auth_code=new_auth,
                                job_status=new_status,
                                waiting_time_raw=new_waiting_raw,
                                comments=new_comments,
                            )
                            st.session_state.edit_selected_job = jn
                            st.session_state.edit_selected_row_id = row_id
                            st.session_state.edit_nonce += 1
                            st.success("Updated.")
                            st.rerun()

        st.divider()
        st.subheader("Records (view only) — strict column order")

        view_df = df.copy().rename(
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

        st.divider()
        st.subheader("Weekly summary (deduped by Date+Job)")

        # ---- FIX: make weekly summary robust to either DB column names OR UI column names ----
        def pick_existing_col(dff: pd.DataFrame, *candidates: str) -> Optional[str]:
            for c in candidates:
                if c in dff.columns:
                    return c
            return None

        dfw = report_df.copy()

        # Exclude Withdraw rows consistently (works for both naming styles)
        status_col = pick_existing_col(dfw, "job_status", "job status")
        if status_col is not None:
            dfw = dfw[dfw[status_col].astype(str).str.lower() != "withdraw"].copy()

        # Date / week start
        date_col = pick_existing_col(dfw, "work_date", "Date")
        if date_col is None:
            st.error("Weekly summary error: no date column found (expected 'work_date' or 'Date').")
            st.stop()

        # Ensure it's actual date objects
        if date_col == "Date":
            dfw["_work_date_for_week"] = to_clean_date_series(dfw["Date"])
        else:
            dfw["_work_date_for_week"] = dfw["work_date"]

        dfw["_work_date_for_week"] = to_clean_date_series(dfw["_work_date_for_week"])
        dfw = dfw[dfw["_work_date_for_week"].notna()].copy()
        dfw["week_start"] = dfw["_work_date_for_week"].apply(week_start)

        # Money columns: accept either DB or UI naming
        amount_col = pick_existing_col(dfw, "amount", "job amount", "job_amount")
        exp_col = pick_existing_col(dfw, "expenses_amount", "expenses Amount", "expenses amount")
        wait_col = pick_existing_col(dfw, "waiting_amount", "waiting owed", "waiting_owed", "waiting amount")

        # If any are missing, create them as zeros so groupby never crashes
        if amount_col is None:
            dfw["_job_amount_for_week"] = 0.0
            amount_col = "_job_amount_for_week"
        if exp_col is None:
            dfw["_expenses_amount_for_week"] = 0.0
            exp_col = "_expenses_amount_for_week"
        if wait_col is None:
            dfw["_waiting_owed_for_week"] = 0.0
            wait_col = "_waiting_owed_for_week"

        dfw[amount_col] = pd.to_numeric(dfw[amount_col], errors="coerce").fillna(0)
        dfw[exp_col] = pd.to_numeric(dfw[exp_col], errors="coerce").fillna(0)
        dfw[wait_col] = pd.to_numeric(dfw[wait_col], errors="coerce").fillna(0)

        weekly = (
            dfw.groupby("week_start", as_index=False)
            .agg(
                rows=("id", "count") if "id" in dfw.columns else (amount_col, "count"),
                job_amount=(amount_col, "sum"),
                expenses_amount=(exp_col, "sum"),
                waiting_owed=(wait_col, "sum"),
            )
            .sort_values("week_start", ascending=False)
        )

        for c in ["job_amount", "expenses_amount", "waiting_owed"]:
            weekly[c] = pd.to_numeric(weekly[c], errors="coerce").fillna(0)

        weekly["total_owed"] = weekly["job_amount"] + weekly["waiting_owed"] + weekly["expenses_amount"]
        st.dataframe(weekly, use_container_width=True, hide_index=True)

        csv_bytes = view_df[UI_COLUMNS].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv_bytes,
            file_name="worklog_filtered.csv",
            mime="text/csv",
        )
>>>>>>> 30ef248 (Fix weekly summary column KeyError)
