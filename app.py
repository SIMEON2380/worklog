import os
import re
import sqlite3
from datetime import date, timedelta
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
APP_TITLE = "Worklog"
DB_DIR = "/var/lib/worklog"
DB_PATH = os.path.join(DB_DIR, "worklog.db")
TABLE_NAME = "work_logs"

WAITING_RATE = 7.50

STATUS_OPTIONS = ["Start", "Completed", "Aborted", "Paid", "Pending", "Withdraw"]
JOB_TYPE_OPTIONS = ["STRD Trade Plate", "Inspect and Collect", "Inspect and Collect 2"]

# REQUIRED BY YOU
JOB_EXPENSE_OPTIONS = ["uber", "taxi", "train", "toll", "other"]

# Your required strict order + titles (do not change)
# You asked to "create a column in the table for waiting time" so it's appended.
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
]

# =========================
# DB helpers
# =========================
def ensure_db_dir():
    os.makedirs(DB_DIR, exist_ok=True)


def get_conn():
    ensure_db_dir()
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def table_columns(conn: sqlite3.Connection, table_name: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows}


def ensure_schema():
    """
    Create table if missing, and migrate if older DB exists.
    Keeps legacy columns so old data stays intact.
    Adds new columns required by your new table structure.
    """
    with get_conn() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- legacy base
                work_date TEXT,
                description TEXT,      -- legacy (kept)
                hours REAL,            -- legacy (hidden)
                amount REAL,           -- used as job amount
                job_id TEXT,           -- job number
                category TEXT,         -- used as job type
                job_status TEXT,

                -- waiting (MAINTAINED)
                waiting_time TEXT,
                waiting_hours REAL,
                waiting_amount REAL,

                -- NEW required fields
                vehicle_description TEXT,
                vehicle_reg TEXT,
                collection_from TEXT,
                delivery_to TEXT,
                job_expenses TEXT,
                expenses_amount REAL,
                auth_code TEXT,

                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()

        cols = table_columns(conn, TABLE_NAME)

        migrations = [
            # legacy + waiting
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
            ("created_at", "TEXT"),
            # new
            ("vehicle_description", "TEXT"),
            ("vehicle_reg", "TEXT"),
            ("collection_from", "TEXT"),
            ("delivery_to", "TEXT"),
            ("job_expenses", "TEXT"),
            ("expenses_amount", "REAL"),
            ("auth_code", "TEXT"),
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
# Parsing utilities
# =========================
def to_clean_date_series(s: pd.Series) -> pd.Series:
    """Coerce messy date column into python datetime.date values."""
    if s is None:
        return pd.Series(dtype="object")

    s2 = s.copy()

    if pd.api.types.is_datetime64_any_dtype(s2):
        return s2.dt.date

    numeric_mask = (
        pd.to_numeric(s2, errors="coerce").notna()
        & s2.astype(str).str.match(r"^\s*\d+(\.\d+)?\s*$", na=False)
    )
    out = pd.Series([pd.NaT] * len(s2), index=s2.index, dtype="object")

    if numeric_mask.any():
        nums = pd.to_numeric(s2[numeric_mask], errors="coerce")
        dt_nums = pd.to_datetime(nums, unit="D", origin="1899-12-30", errors="coerce")
        out.loc[numeric_mask] = dt_nums.dt.date

    rest_mask = ~numeric_mask
    if rest_mask.any():
        dt_rest = pd.to_datetime(s2[rest_mask], errors="coerce", dayfirst=True)
        out.loc[rest_mask] = dt_rest.dt.date

    return out


def safe_date_bounds(dates: Optional[pd.Series]) -> Tuple[date, date]:
    today = date.today()
    if dates is None or len(dates) == 0:
        return today, today
    dt = pd.to_datetime(dates, errors="coerce")
    if dt.notna().any():
        return dt.min().date(), dt.max().date()
    return today, today


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def parse_waiting_time(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Accepts:
      - "10-12"
      - "10:30-12:15"
      - "9.5-11" => 09:30-11:00
    Returns (hours, normalized "HH:MM-HH:MM") or (None, None).
    """
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
        diff += 24 * 60  # crosses midnight

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

    if df.empty:
        return df

    df["work_date"] = to_clean_date_series(df.get("work_date"))
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")
    df["expenses_amount"] = pd.to_numeric(df.get("expenses_amount"), errors="coerce")

    df["waiting_time"] = df.get("waiting_time", "").fillna("").astype(str)
    df["waiting_hours"] = pd.to_numeric(df.get("waiting_hours"), errors="coerce")
    df["waiting_amount"] = pd.to_numeric(df.get("waiting_amount"), errors="coerce")

    df["job_id"] = df.get("job_id", "").fillna("").astype(str)
    df["category"] = df.get("category", "").fillna("").astype(str)

    df["vehicle_description"] = df.get("vehicle_description", "").fillna("").astype(str)
    df["vehicle_reg"] = df.get("vehicle_reg", "").fillna("").astype(str)
    df["collection_from"] = df.get("collection_from", "").fillna("").astype(str)
    df["delivery_to"] = df.get("delivery_to", "").fillna("").astype(str)
    df["job_expenses"] = df.get("job_expenses", "").fillna("").astype(str)
    df["auth_code"] = df.get("auth_code", "").fillna("").astype(str)

    df["job_status"] = df.get("job_status", "").fillna("").astype(str)
    df["job_status"] = df["job_status"].apply(lambda x: x if x in STATUS_OPTIONS else "Pending")

    return df


def read_row_by_id(row_id: int) -> Optional[dict]:
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(f"SELECT * FROM {TABLE_NAME} WHERE id = ?", (row_id,)).fetchone()
        return dict(row) if row else None


def read_rows_by_job_number(job_number: str) -> pd.DataFrame:
    job_number = str(job_number).strip()
    if not job_number:
        return pd.DataFrame()

    with get_conn() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT * FROM {TABLE_NAME}
            WHERE TRIM(COALESCE(job_id,'')) = ?
            ORDER BY work_date DESC, id DESC
            """,
            conn,
            params=(job_number,),
        )

    if df.empty:
        return df

    df["work_date"] = to_clean_date_series(df.get("work_date"))
    return df


def insert_row(
    work_date_val: date,
    job_number: str,
    job_type: str,
    vehicle_description: str,
    vehicle_reg: str,
    collection_from: str,
    delivery_to: str,
    job_amount: Optional[float],
    job_expenses: str,
    expenses_amount: Optional[float],
    auth_code: str,
    job_status: str,
    waiting_time_raw: str,
):
    w_hours, w_norm = parse_waiting_time(waiting_time_raw)
    w_amount = (w_hours * WAITING_RATE) if (w_hours is not None) else None

    wd = work_date_val.isoformat()
    status = job_status if job_status in STATUS_OPTIONS else "Pending"
    jt = job_type if job_type in JOB_TYPE_OPTIONS else JOB_TYPE_OPTIONS[0]
    je = job_expenses if job_expenses in JOB_EXPENSE_OPTIONS else JOB_EXPENSE_OPTIONS[-1]  # other

    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME}
              (
                work_date,
                job_id,
                category,
                vehicle_description,
                vehicle_reg,
                collection_from,
                delivery_to,
                amount,
                job_expenses,
                expenses_amount,
                auth_code,
                job_status,

                waiting_time,
                waiting_hours,
                waiting_amount,

                -- legacy fields kept
                description,
                hours
              )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                wd,
                str(job_number).strip(),
                jt,
                str(vehicle_description or "").strip(),
                str(vehicle_reg or "").strip(),
                str(collection_from or "").strip(),
                str(delivery_to or "").strip(),
                float(job_amount) if job_amount is not None else None,
                je,
                float(expenses_amount) if expenses_amount is not None else None,
                str(auth_code or "").strip(),
                status,
                w_norm or "",
                w_hours,
                w_amount,
                "",   # legacy description
                None  # legacy hours
            ),
        )
        conn.commit()


def update_row_by_id(
    row_id: int,
    work_date_val: date,
    job_number: str,
    job_type: str,
    vehicle_description: str,
    vehicle_reg: str,
    collection_from: str,
    delivery_to: str,
    job_amount: Optional[float],
    job_expenses: str,
    expenses_amount: Optional[float],
    auth_code: str,
    job_status: str,
    waiting_time_raw: str,
):
    w_hours, w_norm = parse_waiting_time(waiting_time_raw)
    w_amount = (w_hours * WAITING_RATE) if (w_hours is not None) else None

    wd = work_date_val.isoformat()
    status = job_status if job_status in STATUS_OPTIONS else "Pending"
    jt = job_type if job_type in JOB_TYPE_OPTIONS else JOB_TYPE_OPTIONS[0]
    je = job_expenses if job_expenses in JOB_EXPENSE_OPTIONS else JOB_EXPENSE_OPTIONS[-1]

    with get_conn() as conn:
        conn.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET
                work_date = ?,
                job_id = ?,
                category = ?,
                vehicle_description = ?,
                vehicle_reg = ?,
                collection_from = ?,
                delivery_to = ?,
                amount = ?,
                job_expenses = ?,
                expenses_amount = ?,
                auth_code = ?,
                job_status = ?,

                waiting_time = ?,
                waiting_hours = ?,
                waiting_amount = ?
            WHERE id = ?
            """,
            (
                wd,
                str(job_number).strip(),
                jt,
                str(vehicle_description or "").strip(),
                str(vehicle_reg or "").strip(),
                str(collection_from or "").strip(),
                str(delivery_to or "").strip(),
                float(job_amount) if job_amount is not None else None,
                je,
                float(expenses_amount) if expenses_amount is not None else None,
                str(auth_code or "").strip(),
                status,
                w_norm or "",
                w_hours,
                w_amount,
                int(row_id),
            ),
        )
        conn.commit()


def insert_many(df: pd.DataFrame) -> int:
    """
    Import Excel/CSV with flexible column names.

    Required:
      - Date / work_date
      - job number / job_number / job_id

    Optional:
      - job type / category
      - vehcile description / vehicle_description
      - vehicle Reg / vehicle_reg
      - collection from / collection_from
      - delivery to / delivery_to
      - job amount / amount
      - Job Expenses / job_expenses
      - expenses Amount / expenses_amount
      - Auth code / auth_code
      - job status / job_status
      - waiting time / waiting_time
    """
    if df is None or df.empty:
        return 0

    df2 = df.copy()
    cols_lower = {str(c).lower().strip(): c for c in df2.columns}

    def pick(*names):
        for n in names:
            key = str(n).lower().strip()
            if key in cols_lower:
                return cols_lower[key]
        return None

    c_date = pick("date", "work_date")
    c_job = pick("job number", "job_number", "job_id", "jobid", "job")

    if c_date is None:
        raise ValueError("Upload must include Date (or work_date).")
    if c_job is None:
        raise ValueError("Upload must include job number (or job_id/job_number).")

    c_job_type = pick("job type", "job_type", "category")
    c_vdesc = pick("vehcile description", "vehicle description", "vehicle_description")
    c_vreg = pick("vehicle reg", "vehicle Reg", "vehicle_reg")
    c_from = pick("collection from", "collection_from")
    c_to = pick("delivery to", "delivery_to")
    c_amt = pick("job amount", "job_amount", "amount")
    c_exp_type = pick("job expenses", "Job Expenses", "job_expenses")
    c_exp_amt = pick("expenses amount", "expenses Amount", "expenses_amount")
    c_auth = pick("auth code", "Auth code", "auth_code")
    c_status = pick("job status", "job_status", "status")
    c_waiting = pick("waiting time", "waiting_time", "waiting")

    df2["work_date"] = to_clean_date_series(df2[c_date])
    df2["job_id"] = df2[c_job].fillna("").astype(str).str.strip()

    df2["job_type"] = df2[c_job_type].fillna("").astype(str).str.strip() if c_job_type else JOB_TYPE_OPTIONS[0]
    df2["job_type"] = df2["job_type"].apply(lambda x: x if x in JOB_TYPE_OPTIONS else JOB_TYPE_OPTIONS[0])

    df2["vehicle_description"] = df2[c_vdesc].fillna("").astype(str).str.strip() if c_vdesc else ""
    df2["vehicle_reg"] = df2[c_vreg].fillna("").astype(str).str.strip() if c_vreg else ""
    df2["collection_from"] = df2[c_from].fillna("").astype(str).str.strip() if c_from else ""
    df2["delivery_to"] = df2[c_to].fillna("").astype(str).str.strip() if c_to else ""

    df2["amount"] = pd.to_numeric(df2[c_amt], errors="coerce") if c_amt else None

    df2["job_expenses"] = df2[c_exp_type].fillna("other").astype(str).str.strip().str.lower() if c_exp_type else "other"
    df2["job_expenses"] = df2["job_expenses"].apply(lambda x: x if x in JOB_EXPENSE_OPTIONS else "other")

    df2["expenses_amount"] = pd.to_numeric(df2[c_exp_amt], errors="coerce") if c_exp_amt else None
    df2["auth_code"] = df2[c_auth].fillna("").astype(str).str.strip() if c_auth else ""

    df2["job_status"] = df2[c_status].fillna("Pending").astype(str).str.strip() if c_status else "Pending"
    df2["job_status"] = df2["job_status"].apply(lambda x: x if x in STATUS_OPTIONS else "Pending")

    df2["waiting_raw"] = df2[c_waiting].fillna("").astype(str) if c_waiting else ""

    wh_list, wn_list, wa_list = [], [], []
    for txt in df2["waiting_raw"].tolist():
        wh, wn = parse_waiting_time(txt)
        wh_list.append(wh)
        wn_list.append(wn or "")
        wa_list.append((wh * WAITING_RATE) if wh is not None else None)

    df2["waiting_hours"] = wh_list
    df2["waiting_time"] = wn_list
    df2["waiting_amount"] = wa_list

    # enforce required
    df2 = df2[df2["work_date"].notna()].copy()
    df2 = df2[df2["job_id"] != ""].copy()
    if df2.empty:
        return 0

    df2["work_date"] = df2["work_date"].apply(lambda d: d.isoformat() if isinstance(d, date) else None)

    rows = list(
        zip(
            df2["work_date"].tolist(),
            df2["job_id"].tolist(),
            df2["job_type"].tolist(),
            df2["vehicle_description"].tolist(),
            df2["vehicle_reg"].tolist(),
            df2["collection_from"].tolist(),
            df2["delivery_to"].tolist(),
            df2["amount"].tolist(),
            df2["job_expenses"].tolist(),
            df2["expenses_amount"].tolist(),
            df2["auth_code"].tolist(),
            df2["job_status"].tolist(),
            df2["waiting_time"].tolist(),
            df2["waiting_hours"].tolist(),
            df2["waiting_amount"].tolist(),
        )
    )

    with get_conn() as conn:
        conn.executemany(
            f"""
            INSERT INTO {TABLE_NAME}
              (
                work_date,
                job_id,
                category,
                vehicle_description,
                vehicle_reg,
                collection_from,
                delivery_to,
                amount,
                job_expenses,
                expenses_amount,
                auth_code,
                job_status,
                waiting_time,
                waiting_hours,
                waiting_amount,
                description,
                hours
              )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    wd,
                    job_id,
                    jt,
                    vdesc,
                    vreg,
                    cfrom,
                    cto,
                    amt,
                    je,
                    exp_amt,
                    auth,
                    status,
                    wtime,
                    wh,
                    wa,
                    "",    # legacy description
                    None,  # legacy hours
                )
                for (
                    wd,
                    job_id,
                    jt,
                    vdesc,
                    vreg,
                    cfrom,
                    cto,
                    amt,
                    je,
                    exp_amt,
                    auth,
                    status,
                    wtime,
                    wh,
                    wa,
                ) in rows
            ],
        )
        conn.commit()

    return len(rows)


def delete_duplicates():
    with get_conn() as conn:
        conn.execute(
            f"""
            DELETE FROM {TABLE_NAME}
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM {TABLE_NAME}
                GROUP BY
                    COALESCE(work_date,''),
                    COALESCE(job_id,''),
                    COALESCE(category,''),
                    COALESCE(vehicle_description,''),
                    COALESCE(vehicle_reg,''),
                    COALESCE(collection_from,''),
                    COALESCE(delivery_to,''),
                    COALESCE(amount, -999999),
                    COALESCE(job_expenses,''),
                    COALESCE(expenses_amount, -999999),
                    COALESCE(auth_code,''),
                    COALESCE(job_status,''),
                    COALESCE(waiting_time,'')
            )
            """
        )
        conn.commit()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

ensure_schema()

with st.sidebar:
    st.header("Actions")
    if st.button("Refresh data"):
        st.rerun()
    if st.button("Delete exact duplicates"):
        delete_duplicates()
        st.success("Duplicates removed.")
        st.rerun()

df_all = read_all()
min_d, max_d = safe_date_bounds(df_all["work_date"] if not df_all.empty else None)

with st.sidebar:
    st.header("Filters")
    start_d, end_d = st.date_input(
        "Work date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )
    status_filter = st.multiselect("Job status", options=STATUS_OPTIONS, default=STATUS_OPTIONS)
    search_txt = st.text_input(
        "Search (job number / vehicle reg / auth code / locations)",
        value="",
    ).strip()

df = df_all.copy()
if not df.empty:
    df = df[df["work_date"].notna()].copy()
    df = df[(df["work_date"] >= start_d) & (df["work_date"] <= end_d)]
    df = df[df["job_status"].isin(status_filter)]
    if search_txt:
        mask = (
            df["job_id"].astype(str).str.contains(search_txt, case=False, na=False)
            | df["vehicle_reg"].astype(str).str.contains(search_txt, case=False, na=False)
            | df["auth_code"].astype(str).str.contains(search_txt, case=False, na=False)
            | df["collection_from"].astype(str).str.contains(search_txt, case=False, na=False)
            | df["delivery_to"].astype(str).str.contains(search_txt, case=False, na=False)
        )
        df = df[mask]

tab1, tab2, tab3 = st.tabs(["Add entry", "Upload Excel/CSV", "View & Edit"])

# -------- Add entry --------
with tab1:
    st.subheader("Add entry")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        work_date_val = st.date_input("Date", value=date.today())
        job_number = st.text_input("job number (required)")
        job_type = st.selectbox("job type", JOB_TYPE_OPTIONS, index=0)
        job_status = st.selectbox("job status", STATUS_OPTIONS, index=STATUS_OPTIONS.index("Pending"))
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
        waiting_time_raw = st.text_input("waiting time (e.g. 10-12 or 10:30-12:15)", value="")

        w_hours, w_norm = parse_waiting_time(waiting_time_raw)
        if waiting_time_raw.strip():
            if w_hours is None:
                st.error("Waiting time format invalid. Use like 10-12 or 10:30-12:15.")
            else:
                st.write(
                    f"Waiting: **{w_norm}** | Hours: **{w_hours:.2f}** | Owed: **£{(w_hours*WAITING_RATE):.2f}**"
                )

    if st.button("Save entry"):
        if str(job_number).strip() == "":
            st.error("job number is required.")
        elif waiting_time_raw.strip() and w_hours is None:
            st.error("Fix waiting time format before saving.")
        else:
            insert_row(
                work_date_val=work_date_val,
                job_number=job_number,
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
            )
            st.success("Saved.")
            st.rerun()

# -------- Upload --------
with tab2:
    st.subheader("Upload Excel/CSV")
    st.write(
        "Required columns: **Date** (or `work_date`) and **job number** (or `job_id`/`job_number`). "
        "Optional: job type, vehicle fields, job amount, job expenses, expenses amount, auth code, job status, waiting time."
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
        # KPIs
        total_job = df["amount"].fillna(0).sum()
        total_exp = df["expenses_amount"].fillna(0).sum()
        total_wait = df["waiting_amount"].fillna(0).sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", f"{len(df):,}")
        k2.metric("Job amount", f"{total_job:,.2f}")
        k3.metric("Expenses amount", f"{total_exp:,.2f}")
        k4.metric("Waiting owed", f"{total_wait:,.2f}")

        st.divider()
        st.markdown("## Edit (select a job number and save)")
        st.caption("Editing is driven by **job number**. If a job number has multiple rows, pick the exact one.")

        job_numbers = sorted([jn for jn in df["job_id"].dropna().astype(str).str.strip().unique().tolist() if jn != ""])
        selected_job_number = st.selectbox("Select job number", options=job_numbers)

        matches = read_rows_by_job_number(selected_job_number)
        if matches.empty:
            st.error("Could not load that job number from the database.")
        else:
            if len(matches) > 1:
                m2 = matches.copy()
                m2["pick_label"] = m2.apply(
                    lambda r: f"{r.get('work_date')} | id {r.get('id')} | {str(r.get('vehicle_reg') or '').strip()} | {str(r.get('auth_code') or '').strip()}",
                    axis=1,
                )
                pick_label = st.selectbox("This job number has multiple rows — pick one", options=m2["pick_label"].tolist())
                picked_row = m2[m2["pick_label"] == pick_label].iloc[0].to_dict()
            else:
                picked_row = matches.iloc[0].to_dict()

            row_id = int(picked_row["id"])

            cur_date_series = to_clean_date_series(pd.Series([picked_row.get("work_date")]))
            cur_date = cur_date_series.iloc[0] if isinstance(cur_date_series.iloc[0], date) else date.today()

            cur_job_number = str(picked_row.get("job_id") or "").strip()

            cur_job_type = str(picked_row.get("category") or "").strip()
            if cur_job_type not in JOB_TYPE_OPTIONS:
                cur_job_type = JOB_TYPE_OPTIONS[0]

            cur_vdesc = str(picked_row.get("vehicle_description") or "").strip()
            cur_vreg = str(picked_row.get("vehicle_reg") or "").strip()
            cur_from = str(picked_row.get("collection_from") or "").strip()
            cur_to = str(picked_row.get("delivery_to") or "").strip()

            cur_job_amount = picked_row.get("amount")
            cur_job_amount = float(cur_job_amount) if cur_job_amount is not None else 0.0

            cur_job_exp = str(picked_row.get("job_expenses") or "").strip().lower()
            if cur_job_exp not in JOB_EXPENSE_OPTIONS:
                cur_job_exp = "other"

            cur_exp_amt = picked_row.get("expenses_amount")
            cur_exp_amt = float(cur_exp_amt) if cur_exp_amt is not None else 0.0

            cur_auth = str(picked_row.get("auth_code") or "").strip()

            cur_status = str(picked_row.get("job_status") or "Pending").strip()
            if cur_status not in STATUS_OPTIONS:
                cur_status = "Pending"

            cur_waiting = str(picked_row.get("waiting_time") or "").strip()

            with st.form("edit_form"):
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    new_date = st.date_input("Date", value=cur_date)
                    new_job_number = st.text_input("job number (required)", value=cur_job_number)
                    new_job_type = st.selectbox("job type", JOB_TYPE_OPTIONS, index=JOB_TYPE_OPTIONS.index(cur_job_type))
                    new_status = st.selectbox("job status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(cur_status))

                with c2:
                    new_vdesc = st.text_input("vehcile description", value=cur_vdesc)
                    new_vreg = st.text_input("vehicle Reg", value=cur_vreg)

                with c3:
                    new_from = st.text_input("collection from", value=cur_from)
                    new_to = st.text_input("delivery to", value=cur_to)

                with c4:
                    new_job_amount = st.number_input("job amount", step=0.5, value=float(cur_job_amount))
                    new_job_expenses = st.selectbox("Job Expenses", JOB_EXPENSE_OPTIONS, index=JOB_EXPENSE_OPTIONS.index(cur_job_exp))
                    new_exp_amt = st.number_input("expenses Amount", step=0.5, value=float(cur_exp_amt))
                    new_auth = st.text_input("Auth code", value=cur_auth)

                    new_waiting_raw = st.text_input("waiting time", value=cur_waiting)
                    wh, wn = parse_waiting_time(new_waiting_raw)
                    if new_waiting_raw.strip():
                        if wh is None:
                            st.error("Waiting time format invalid. Use like 10-12 or 10:30-12:15.")
                        else:
                            st.write(f"Waiting: **{wn}** | Hours: **{wh:.2f}** | Owed: **£{(wh*WAITING_RATE):.2f}**")

                save = st.form_submit_button("Save changes")

                if save:
                    if str(new_job_number).strip() == "":
                        st.error("job number is required.")
                    elif new_waiting_raw.strip() and wh is None:
                        st.error("Fix waiting time format before saving.")
                    else:
                        update_row_by_id(
                            row_id=row_id,
                            work_date_val=new_date,
                            job_number=new_job_number,
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
                        )
                        st.success("Updated.")
                        st.rerun()

        st.divider()
        st.subheader("Records (view only) — strict column order")

        view_df = df.copy()
        view_df = view_df.rename(
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
            }
        )

        for col in UI_COLUMNS:
            if col not in view_df.columns:
                view_df[col] = ""

        st.dataframe(
            view_df[UI_COLUMNS],
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("Weekly summary")
        dfw = df.copy()
        dfw["week_start"] = dfw["work_date"].apply(week_start)

        weekly = (
            dfw.groupby("week_start", as_index=False)
            .agg(
                rows=("id", "count"),
                job_amount=("amount", "sum"),
                expenses_amount=("expenses_amount", "sum"),
                waiting_owed=("waiting_amount", "sum"),
            )
            .sort_values("week_start", ascending=False)
        )

        for c in ["job_amount", "expenses_amount", "waiting_owed"]:
            weekly[c] = weekly[c].fillna(0)
        weekly["total_owed"] = weekly["job_amount"] + weekly["waiting_owed"]

        st.dataframe(weekly, use_container_width=True, hide_index=True)

        csv_bytes = view_df[UI_COLUMNS].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv_bytes,
            file_name="worklog_filtered.csv",
            mime="text/csv",
        )
