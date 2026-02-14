import os
import re
import sqlite3
from datetime import date, datetime, timedelta
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
    # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
    return {r[1] for r in rows}


def ensure_schema():
    """
    Create table if missing, and migrate (ALTER TABLE add columns) if older DB exists.
    """
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

                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()

        cols = table_columns(conn, TABLE_NAME)

        # Migrations for older DBs
        migrations = [
            ("job_status", "TEXT"),
            ("waiting_time", "TEXT"),
            ("waiting_hours", "REAL"),
            ("waiting_amount", "REAL"),
        ]

        for col, col_type in migrations:
            if col not in cols:
                conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {col} {col_type}")
        conn.commit()

        # Indexes
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_work_date ON {TABLE_NAME}(work_date)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_job_id ON {TABLE_NAME}(job_id)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_status ON {TABLE_NAME}(job_status)")
        conn.commit()


# =========================
# Date + parsing utilities
# =========================
def to_clean_date_series(s: pd.Series) -> pd.Series:
    """
    Convert a messy 'work_date' column into python datetime.date values.
    Handles:
      - normal date strings
      - pandas timestamps
      - Excel serial numbers
      - NaN / blanks
    Returns a Series of dtype object containing datetime.date or NaT.
    """
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


def safe_date_bounds(dates: pd.Series) -> Tuple[date, date]:
    if dates is None or len(dates) == 0:
        today = date.today()
        return today, today

    dt = pd.to_datetime(dates, errors="coerce")
    if dt.notna().any():
        return dt.min().date(), dt.max().date()

    today = date.today()
    return today, today


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def parse_waiting_time(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Accepts formats like:
      - "10-12"
      - "10:30-12:15"
      - "10.5-12" (treat 10.5 as 10:30)
      - "10 - 12"
    Returns (hours, normalized_string) or (None, None) if blank/invalid.
    If end < start, assumes it crosses midnight and adds 24h.
    """
    if text is None:
        return None, None

    raw = str(text).strip()
    if raw == "":
        return None, None

    # Normalize separators
    raw = raw.replace("–", "-").replace("—", "-")
    raw = re.sub(r"\s+", "", raw)

    # Must contain exactly one dash
    if raw.count("-") != 1:
        return None, None

    start_s, end_s = raw.split("-")

    def to_minutes(t: str) -> Optional[int]:
        # "10" -> 10:00
        # "10:30" -> 10:30
        # "10.5" -> 10:30
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

        # decimal hours like 10.5
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

    # Normalized display "HH:MM-HH:MM"
    def fmt(mins: int) -> str:
        mins = mins % (24 * 60)
        h = mins // 60
        m = mins % 60
        return f"{h:02d}:{m:02d}"

    norm = f"{fmt(start_min)}-{fmt(end_min)}"
    return hours, norm


# =========================
# Data access
# =========================
def read_all() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY work_date DESC, id DESC", conn)

    if not df.empty:
        df["work_date"] = to_clean_date_series(df["work_date"])
        df["hours"] = pd.to_numeric(df.get("hours"), errors="coerce")
        df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")
        df["waiting_hours"] = pd.to_numeric(df.get("waiting_hours"), errors="coerce")
        df["waiting_amount"] = pd.to_numeric(df.get("waiting_amount"), errors="coerce")
        df["job_status"] = df.get("job_status", "").fillna("").astype(str)
        df["waiting_time"] = df.get("waiting_time", "").fillna("").astype(str)
    return df


def insert_row(
    work_date_val: date,
    description: str,
    hours: Optional[float],
    amount: Optional[float],
    job_id: str,
    category: str,
    job_status: str,
    waiting_time_raw: str,
):
    w_hours, w_norm = parse_waiting_time(waiting_time_raw)
    w_amount = (w_hours * WAITING_RATE) if (w_hours is not None) else None

    wd = work_date_val.isoformat() if isinstance(work_date_val, date) else None
    status = job_status if job_status in STATUS_OPTIONS else "Pending"

    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME}
              (work_date, description, hours, amount, job_id, category, job_status, waiting_time, waiting_hours, waiting_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (wd, description, hours, amount, job_id, category, status, w_norm, w_hours, w_amount),
        )
        conn.commit()


def insert_many(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    df2 = df.copy()

    cols_lower = {c.lower(): c for c in df2.columns}

    def pick(*names):
        for n in names:
            if n in cols_lower:
                return cols_lower[n]
        return None

    c_work_date = pick("work_date", "date")
    if c_work_date is None:
        raise ValueError("Upload must include a 'work_date' (or 'date') column.")

    c_desc = pick("description", "desc", "details")
    c_hours = pick("hours", "hour")
    c_amount = pick("amount", "cost", "value")
    c_job_id = pick("job_id", "job", "jobid")
    c_category = pick("category", "type")

    c_status = pick("job_status", "status")
    c_waiting_time = pick("waiting_time", "waiting", "waitingtime")

    df2["work_date"] = to_clean_date_series(df2[c_work_date])
    df2["description"] = df2[c_desc] if c_desc else ""
    df2["hours"] = pd.to_numeric(df2[c_hours], errors="coerce") if c_hours else None
    df2["amount"] = pd.to_numeric(df2[c_amount], errors="coerce") if c_amount else None
    df2["job_id"] = df2[c_job_id].astype(str) if c_job_id else ""
    df2["category"] = df2[c_category].astype(str) if c_category else ""

    # Status
    if c_status:
        df2["job_status"] = df2[c_status].fillna("").astype(str).str.strip()
    else:
        df2["job_status"] = "Pending"
    df2["job_status"] = df2["job_status"].apply(lambda x: x if x in STATUS_OPTIONS else "Pending")

    # Waiting time -> calculate
    if c_waiting_time:
        df2["waiting_time_raw"] = df2[c_waiting_time].fillna("").astype(str)
    else:
        df2["waiting_time_raw"] = ""

    wh_list = []
    wn_list = []
    wa_list = []
    for txt in df2["waiting_time_raw"].tolist():
        wh, wn = parse_waiting_time(txt)
        wh_list.append(wh)
        wn_list.append(wn)
        wa_list.append((wh * WAITING_RATE) if wh is not None else None)

    df2["waiting_hours"] = wh_list
    df2["waiting_time"] = wn_list
    df2["waiting_amount"] = wa_list

    # Keep only valid dates
    df2 = df2[df2["work_date"].notna()].copy()
    if df2.empty:
        return 0

    df2["work_date"] = df2["work_date"].apply(lambda d: d.isoformat() if isinstance(d, date) else None)

    rows = list(
        zip(
            df2["work_date"].tolist(),
            df2["description"].fillna("").astype(str).tolist(),
            df2["hours"].tolist(),
            df2["amount"].tolist(),
            df2["job_id"].fillna("").astype(str).tolist(),
            df2["category"].fillna("").astype(str).tolist(),
            df2["job_status"].fillna("Pending").astype(str).tolist(),
            df2["waiting_time"].tolist(),
            df2["waiting_hours"].tolist(),
            df2["waiting_amount"].tolist(),
        )
    )

    with get_conn() as conn:
        conn.executemany(
            f"""
            INSERT INTO {TABLE_NAME}
              (work_date, description, hours, amount, job_id, category, job_status, waiting_time, waiting_hours, waiting_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    return len(rows)


def delete_duplicates():
    """
    Remove duplicates based on key columns, keeping lowest id.
    """
    with get_conn() as conn:
        conn.execute(
            f"""
            DELETE FROM {TABLE_NAME}
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM {TABLE_NAME}
                GROUP BY
                    COALESCE(work_date,''),
                    COALESCE(description,''),
                    COALESCE(hours, -999999),
                    COALESCE(amount, -999999),
                    COALESCE(job_id,''),
                    COALESCE(category,''),
                    COALESCE(job_status,''),
                    COALESCE(waiting_time,''),
                    COALESCE(waiting_hours, -999999),
                    COALESCE(waiting_amount, -999999)
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

# Date range filter
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
    search_txt = st.text_input("Search (description / job / category)", value="").strip()

df = df_all.copy()
if not df.empty:
    df = df[df["work_date"].notna()].copy()
    df = df[(df["work_date"] >= start_d) & (df["work_date"] <= end_d)]
    df = df[df["job_status"].isin(status_filter)]

    if search_txt:
        mask = (
            df["description"].fillna("").str.contains(search_txt, case=False, na=False)
            | df["job_id"].fillna("").astype(str).str.contains(search_txt, case=False, na=False)
            | df["category"].fillna("").astype(str).str.contains(search_txt, case=False, na=False)
        )
        df = df[mask]

tab1, tab2, tab3 = st.tabs(["Add entry", "Upload Excel/CSV", "View & Summary"])

with tab1:
    st.subheader("Add a single entry")

    c1, c2, c3 = st.columns(3)
    with c1:
        work_date_val = st.date_input("Work date", value=date.today())
        job_status = st.selectbox("Job status", STATUS_OPTIONS, index=STATUS_OPTIONS.index("Pending"))
        job_id = st.text_input("Job ID (optional)")
    with c2:
        hours = st.number_input("Hours (work)", min_value=0.0, step=0.5, value=0.0)
        amount = st.number_input("Amount (£) (work)", step=0.5, value=0.0)
        category = st.text_input("Category (optional)")
    with c3:
        waiting_time_raw = st.text_input("Waiting time (e.g. 10-12 or 10:30-12:15)", value="")
        w_hours, w_norm = parse_waiting_time(waiting_time_raw)
        if waiting_time_raw.strip():
            if w_hours is None:
                st.error("Waiting time format invalid. Use like 10-12 or 10:30-12:15.")
            else:
                st.write(f"Normalized: **{w_norm}**")
                st.write(f"Waiting hours: **{w_hours:.2f}**")
                st.write(f"Waiting amount owed (£{WAITING_RATE:.2f}/hr): **£{(w_hours*WAITING_RATE):.2f}**")

    description = st.text_area("Description", height=90)

    if st.button("Save entry"):
        if waiting_time_raw.strip() and w_hours is None:
            st.error("Fix waiting time format before saving.")
        else:
            insert_row(
                work_date_val=work_date_val,
                description=description,
                hours=hours,
                amount=amount,
                job_id=job_id,
                category=category,
                job_status=job_status,
                waiting_time_raw=waiting_time_raw,
            )
            st.success("Saved.")
            st.rerun()

with tab2:
    st.subheader("Upload file (Excel or CSV)")
    st.write(
        "Expected columns: `work_date` (or `date`), and optional `description`, `hours`, `amount`, `job_id`, "
        "`category`, `job_status` (or `status`), `waiting_time`."
    )
    st.caption("Waiting time examples: 10-12, 10:30-12:15, 9.5-11 (means 09:30-11:00).")

    up = st.file_uploader("Choose a file", type=["xlsx", "xls", "csv"])
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                up_df = pd.read_csv(up)
            else:
                up_df = pd.read_excel(up)

            st.write("Preview:")
            st.dataframe(up_df.head(30), use_container_width=True)

            if st.button("Import into database"):
                n = insert_many(up_df)
                st.success(f"Imported {n} rows.")
                st.rerun()

        except Exception as e:
            st.error(f"Upload/import failed: {e}")

with tab3:
    st.subheader("Records")
    st.caption(f"Database: {DB_PATH}")

    if df.empty:
        st.info("No records in this range.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", f"{len(df):,}")
        k2.metric("Work hours", f"{df['hours'].fillna(0).sum():,.2f}")
        k3.metric("Work amount (£)", f"{df['amount'].fillna(0).sum():,.2f}")
        k4.metric("Waiting owed (£)", f"{df['waiting_amount'].fillna(0).sum():,.2f}")

        st.dataframe(
            df[
                [
                    "id",
                    "work_date",
                    "job_id",
                    "job_status",
                    "category",
                    "hours",
                    "amount",
                    "waiting_time",
                    "waiting_hours",
                    "waiting_amount",
                    "description",
                    "created_at",
                ]
            ],
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
                work_hours=("hours", "sum"),
                work_amount=("amount", "sum"),
                waiting_hours=("waiting_hours", "sum"),
                waiting_owed=("waiting_amount", "sum"),
            )
            .sort_values("week_start", ascending=False)
        )

        for c in ["work_hours", "work_amount", "waiting_hours", "waiting_owed"]:
            weekly[c] = weekly[c].fillna(0)

        st.dataframe(weekly, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv_bytes,
            file_name="worklog_filtered.csv",
            mime="text/csv",
        )
