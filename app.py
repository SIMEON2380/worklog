import os
import re
import sqlite3
import subprocess
from datetime import date, timedelta
from typing import Optional, Tuple, Any, List

import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
APP_TITLE = "Worklog"

# systemd: Environment=WORKLOG_DB_DIR=/var/lib/worklog
DB_DIR = os.environ.get("WORKLOG_DB_DIR", "/var/lib/worklog")
DB_PATH = os.path.join(DB_DIR, "worklog.db")
TABLE_NAME = "work_logs"

WAITING_RATE = 7.50

STATUS_OPTIONS = ["Start", "Completed", "Aborted", "Paid", "Pending", "Withdraw"]
JOB_TYPE_OPTIONS = ["STRD Trade Plate", "Inspect and Collect", "Inspect and Collect 2"]

JOB_EXPENSE_OPTIONS = ["uber", "taxi", "train", "toll", "other"]

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
    "created_at",
]

# =========================
# Helpers
# =========================
def normalize_status(x: Any) -> str:
    s = str(x or "").strip()
    return s if s in STATUS_OPTIONS else "Pending"


def normalize_job_type(x: Any) -> str:
    s = str(x or "").strip()
    return s if s in JOB_TYPE_OPTIONS else JOB_TYPE_OPTIONS[0]


def normalize_expense_type(x: Any) -> str:
    s = str(x or "").strip().lower()
    return s if s in JOB_EXPENSE_OPTIONS else "other"


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

    if df.empty:
        return pd.DataFrame(columns=EXPECTED_DB_COLS)

    for c in EXPECTED_DB_COLS:
        if c not in df.columns:
            df[c] = None

    df["work_date"] = to_clean_date_series(df["work_date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["expenses_amount"] = pd.to_numeric(df["expenses_amount"], errors="coerce")
    df["waiting_amount"] = pd.to_numeric(df["waiting_amount"], errors="coerce")
    df["waiting_hours"] = pd.to_numeric(df["waiting_hours"], errors="coerce")

    df["waiting_time"] = df["waiting_time"].fillna("").astype(str)
    df["job_id"] = df["job_id"].apply(clean_job_number)
    df["category"] = df["category"].fillna("").astype(str).apply(normalize_job_type)
    df["job_status"] = df["job_status"].apply(normalize_status)

    df["vehicle_description"] = df["vehicle_description"].fillna("").astype(str)
    df["vehicle_reg"] = df["vehicle_reg"].fillna("").astype(str)
    df["collection_from"] = df["collection_from"].fillna("").astype(str)
    df["delivery_to"] = df["delivery_to"].fillna("").astype(str)
    df["job_expenses"] = df["job_expenses"].fillna("").astype(str).apply(normalize_expense_type)
    df["auth_code"] = df["auth_code"].fillna("").astype(str)

    return df


def read_rows_by_job_number(job_number: str) -> pd.DataFrame:
    job_number = clean_job_number(job_number)
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

    for c in EXPECTED_DB_COLS:
        if c not in df.columns:
            df[c] = None
    df["work_date"] = to_clean_date_series(df["work_date"])
    df["job_id"] = df["job_id"].apply(clean_job_number)
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
    job_number = clean_job_number(job_number)

    w_hours, w_norm = parse_waiting_time(waiting_time_raw)
    w_amount = (w_hours * WAITING_RATE) if (w_hours is not None) else None

    wd = work_date_val.isoformat()
    status = normalize_status(job_status)
    jt = normalize_job_type(job_type)
    je = normalize_expense_type(job_expenses)

    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME}
              (
                work_date, job_id, category,
                vehicle_description, vehicle_reg, collection_from, delivery_to,
                amount, job_expenses, expenses_amount, auth_code, job_status,
                waiting_time, waiting_hours, waiting_amount,
                description, hours
              )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                wd,
                job_number,
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
                "",
                None,
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
    job_number = clean_job_number(job_number)

    w_hours, w_norm = parse_waiting_time(waiting_time_raw)
    w_amount = (w_hours * WAITING_RATE) if (w_hours is not None) else None

    wd = work_date_val.isoformat()
    status = normalize_status(job_status)
    jt = normalize_job_type(job_type)
    je = normalize_expense_type(job_expenses)

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
                job_number,
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
    df2["job_id"] = df2[c_job].apply(clean_job_number)

    df2["job_type"] = df2[c_job_type].fillna("").astype(str).str.strip() if c_job_type else JOB_TYPE_OPTIONS[0]
    df2["job_type"] = df2["job_type"].apply(normalize_job_type)

    df2["vehicle_description"] = df2[c_vdesc].fillna("").astype(str).str.strip() if c_vdesc else ""
    df2["vehicle_reg"] = df2[c_vreg].fillna("").astype(str).str.strip() if c_vreg else ""
    df2["collection_from"] = df2[c_from].fillna("").astype(str).str.strip() if c_from else ""
    df2["delivery_to"] = df2[c_to].fillna("").astype(str).str.strip() if c_to else ""

    df2["amount"] = pd.to_numeric(df2[c_amt], errors="coerce") if c_amt else None

    df2["job_expenses"] = df2[c_exp_type].fillna("other").astype(str).str.strip() if c_exp_type else "other"
    df2["job_expenses"] = df2["job_expenses"].apply(normalize_expense_type)

    df2["expenses_amount"] = pd.to_numeric(df2[c_exp_amt], errors="coerce") if c_exp_amt else None
    df2["auth_code"] = df2[c_auth].fillna("").astype(str).str.strip() if c_auth else ""

    df2["job_status"] = df2[c_status].fillna("Pending").astype(str).str.strip() if c_status else "Pending"
    df2["job_status"] = df2["job_status"].apply(normalize_status)

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
                work_date, job_id, category,
                vehicle_description, vehicle_reg, collection_from, delivery_to,
                amount, job_expenses, expenses_amount, auth_code, job_status,
                waiting_time, waiting_hours, waiting_amount,
                description, hours
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
                    "",
                    None,
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


# =========================
# Dedupe logic
# =========================
def count_exact_duplicates() -> int:
    with get_conn() as conn:
        row = conn.execute(
            f"""
            SELECT COALESCE(SUM(cnt - 1), 0) AS dupes
            FROM (
                SELECT COUNT(*) AS cnt
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
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()
        return int(row[0] or 0)


def delete_duplicates_exact() -> int:
    with get_conn() as conn:
        before = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
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
        after = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        return int(before - after)


def count_smart_duplicates() -> int:
    """Duplicates by (work_date, job_id) beyond the first row."""
    with get_conn() as conn:
        row = conn.execute(
            f"""
            SELECT COALESCE(SUM(cnt - 1), 0)
            FROM (
                SELECT COUNT(*) AS cnt
                FROM {TABLE_NAME}
                GROUP BY COALESCE(work_date,''), COALESCE(job_id,'')
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()
        return int(row[0] or 0)


def _row_completeness_score_df(df: pd.DataFrame) -> pd.Series:
    """
    Higher score = more complete row.
    We prefer rows with vehicle_reg, locations, amount, status, etc.
    """
    def filled(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.strip().ne("")

    score = pd.Series(0, index=df.index, dtype="int64")
    score += filled(df["vehicle_reg"]).astype(int) * 3
    score += filled(df["vehicle_description"]).astype(int) * 2
    score += filled(df["collection_from"]).astype(int) * 2
    score += filled(df["delivery_to"]).astype(int) * 2
    score += filled(df["auth_code"]).astype(int) * 2
    score += filled(df["job_status"]).astype(int) * 1
    score += pd.to_numeric(df["amount"], errors="coerce").fillna(0).gt(0).astype(int) * 2
    score += pd.to_numeric(df["expenses_amount"], errors="coerce").fillna(0).gt(0).astype(int) * 1
    score += filled(df["waiting_time"]).astype(int) * 1
    return score


def delete_duplicates_smart() -> int:
    """
    Smart dedupe: for each (work_date, job_id) keep the most complete row.
    Deletes the rest.
    """
    with get_conn() as conn:
        df = pd.read_sql_query(
            f"SELECT id, work_date, job_id, category, job_status, vehicle_description, vehicle_reg, collection_from, delivery_to, amount, job_expenses, expenses_amount, auth_code, waiting_time FROM {TABLE_NAME}",
            conn,
        )

    if df.empty:
        return 0

    df["job_id"] = df["job_id"].apply(clean_job_number)
    df["work_date"] = df["work_date"].fillna("").astype(str).str.strip()

    # only groups that have duplicates
    grp = df.groupby(["work_date", "job_id"]).size().reset_index(name="cnt")
    grp = grp[(grp["work_date"] != "") & (grp["job_id"] != "") & (grp["cnt"] > 1)]
    if grp.empty:
        return 0

    df["score"] = _row_completeness_score_df(df)

    # Keep: highest score, tie -> smallest id
    df_sorted = df.sort_values(["work_date", "job_id", "score", "id"], ascending=[True, True, False, True])
    keep = df_sorted.drop_duplicates(["work_date", "job_id"], keep="first")
    keep_ids = set(keep["id"].astype(int).tolist())

    # Delete other ids only within duplicate groups
    dup_keys = set(map(tuple, grp[["work_date", "job_id"]].values.tolist()))
    to_delete = df[df.apply(lambda r: (r["work_date"], r["job_id"]) in dup_keys and int(r["id"]) not in keep_ids, axis=1)]
    del_ids = to_delete["id"].astype(int).tolist()
    if not del_ids:
        return 0

    with get_conn() as conn:
        before = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        conn.executemany(f"DELETE FROM {TABLE_NAME} WHERE id = ?", [(i,) for i in del_ids])
        conn.commit()
        after = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    return int(before - after)


def dedup_for_reporting(df: pd.DataFrame) -> pd.DataFrame:
    """
    For KPI/weekly totals: dedupe by (work_date, job_id) keeping the most complete row.
    """
    if df.empty:
        return df
    d = df.copy()
    d["work_date_str"] = d["work_date"].astype(str)
    d["job_id"] = d["job_id"].apply(clean_job_number)
    d["score"] = _row_completeness_score_df(d)

    d = d.sort_values(["work_date_str", "job_id", "score", "id"], ascending=[True, True, False, True])
    d = d.drop_duplicates(["work_date_str", "job_id"], keep="first")
    return d.drop(columns=["work_date_str", "score"], errors="ignore")


# =========================
# Version
# =========================
def get_live_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

ensure_schema()

if "edit_selected_job" not in st.session_state:
    st.session_state.edit_selected_job = ""
if "edit_selected_row_id" not in st.session_state:
    st.session_state.edit_selected_row_id = None
if "edit_nonce" not in st.session_state:
    st.session_state.edit_nonce = 0

with st.sidebar:
    st.header("Actions")

    exact_dupes = count_exact_duplicates()
    smart_dupes = count_smart_duplicates()

    st.caption(f"Exact duplicates found: {exact_dupes}")
    if st.button("Delete exact duplicates"):
        try:
            deleted = delete_duplicates_exact()
            st.success(f"Deleted {deleted} exact duplicate rows.")
            st.rerun()
        except Exception as e:
            st.error(f"Delete failed: {e}")

    st.caption(f"Duplicates by Date+Job found: {smart_dupes}")
    if st.button("Smart delete duplicates (Date + Job)"):
        try:
            deleted = delete_duplicates_smart()
            st.success(f"Deleted {deleted} duplicates (kept most complete row).")
            st.rerun()
        except Exception as e:
            st.error(f"Smart delete failed: {e}")

    if st.button("Refresh data"):
        st.rerun()

    st.divider()
    st.caption("DB in use:")
    st.code(DB_PATH)
    st.caption(f"Live version: {get_live_version()}")

df_all = read_all()
min_d, max_d = safe_date_bounds(df_all["work_date"] if not df_all.empty else None)
default_end = max(max_d, date.today())

with st.sidebar:
    st.header("Filters")
    date_val = st.date_input(
        "Work date range",
        value=(min_d, default_end),
        min_value=min_d,
        max_value=default_end,
    )
    if isinstance(date_val, (tuple, list)) and len(date_val) == 2:
        start_d, end_d = date_val
    else:
        start_d = date_val
        end_d = date_val

    status_filter = st.multiselect("Job status", options=STATUS_OPTIONS, default=STATUS_OPTIONS)
    search_txt = st.text_input("Search (job / reg / auth / locations)", value="").strip()

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
        work_date_val = st.date_input("Date", value=date.today(), key="add_date")
        job_number = st.text_input("job number (required)", key="add_job_number")
        job_type = st.selectbox("job type", JOB_TYPE_OPTIONS, index=0, key="add_job_type")
        job_status = st.selectbox("job status", STATUS_OPTIONS, index=STATUS_OPTIONS.index("Pending"), key="add_job_status")
    with c2:
        vehicle_description = st.text_input("vehcile description", key="add_vdesc")
        vehicle_reg = st.text_input("vehicle Reg", key="add_vreg")
    with c3:
        collection_from = st.text_input("collection from", key="add_from")
        delivery_to = st.text_input("delivery to", key="add_to")
    with c4:
        job_amount = st.number_input("job amount", step=0.5, value=0.0, key="add_amount")
        job_expenses = st.selectbox("Job Expenses", JOB_EXPENSE_OPTIONS, index=0, key="add_exp_type")
        expenses_amount = st.number_input("expenses Amount", step=0.5, value=0.0, key="add_exp_amt")
        auth_code = st.text_input("Auth code", key="add_auth")
        waiting_time_raw = st.text_input("waiting time (e.g. 10-12 or 10:30-12:15)", value="", key="add_waiting")

        w_hours, w_norm = parse_waiting_time(waiting_time_raw)
        if waiting_time_raw.strip():
            if w_hours is None:
                st.error("Waiting time format invalid. Use like 10-12 or 10:30-12:15.")
            else:
                st.write(f"Waiting: **{w_norm}** | Hours: **{w_hours:.2f}** | Owed: **£{(w_hours*WAITING_RATE):.2f}**")

    if st.button("Save entry"):
        jn = clean_job_number(job_number)
        if jn == "":
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
                    job_amount=float(job_amount) if job_amount is not None else None,
                    job_expenses=job_expenses,
                    expenses_amount=float(expenses_amount) if expenses_amount is not None else None,
                    auth_code=auth_code,
                    job_status=job_status,
                    waiting_time_raw=waiting_time_raw,
                )
                st.success("Saved.")
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")

# -------- Upload --------
with tab2:
    st.subheader("Upload Excel/CSV")
    st.write(
        "Required: **Date** (or `work_date`) and **job number** (or `job_id`/`job_number`). "
        "Optional: job type, vehicle fields, amounts, auth code, status, waiting time."
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
        # IMPORTANT: use deduped rows for totals
        report_df = dedup_for_reporting(df)

        total_job = pd.to_numeric(report_df["amount"], errors="coerce").fillna(0).sum()
        total_exp = pd.to_numeric(report_df["expenses_amount"], errors="coerce").fillna(0).sum()
        total_wait = pd.to_numeric(report_df["waiting_amount"], errors="coerce").fillna(0).sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows (filtered)", f"{len(df):,}")
        k2.metric("Job amount (deduped)", f"£{total_job:,.2f}")
        k3.metric("Expenses (deduped)", f"£{total_exp:,.2f}")
        k4.metric("Waiting owed (deduped)", f"£{total_wait:,.2f}")

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
            else:
                if len(matches) > 1:
                    m2 = matches.copy()
                    m2["work_date_str"] = m2["work_date"].astype(str)
                    m2["pick_label"] = m2.apply(
                        lambda r: f"{r.get('work_date_str')} | id {int(r.get('id'))} | {str(r.get('vehicle_reg') or '').strip()} | {str(r.get('auth_code') or '').strip()}",
                        axis=1,
                    )
                    labels = m2["pick_label"].tolist()
                    default_label = labels[0]
                    if st.session_state.edit_selected_row_id is not None:
                        hit = m2[m2["id"].astype(int) == int(st.session_state.edit_selected_row_id)]
                        if not hit.empty:
                            default_label = hit.iloc[0]["pick_label"]

                    pick_label = st.selectbox(
                        "This job number has multiple rows — pick one",
                        options=labels,
                        index=labels.index(default_label),
                        key="edit_row_pick",
                    )
                    picked_row = m2[m2["pick_label"] == pick_label].iloc[0].to_dict()
                else:
                    picked_row = matches.iloc[0].to_dict()

                row_id = int(picked_row["id"])
                st.session_state.edit_selected_row_id = row_id

                cur_date_series = to_clean_date_series(pd.Series([picked_row.get("work_date")]))
                cur_date = cur_date_series.iloc[0] if isinstance(cur_date_series.iloc[0], date) else date.today()

                cur_job_number = clean_job_number(picked_row.get("job_id"))
                cur_job_type = normalize_job_type(picked_row.get("category"))
                cur_vdesc = str(picked_row.get("vehicle_description") or "").strip()
                cur_vreg = str(picked_row.get("vehicle_reg") or "").strip()
                cur_from = str(picked_row.get("collection_from") or "").strip()
                cur_to = str(picked_row.get("delivery_to") or "").strip()
                cur_job_amount = float(picked_row.get("amount") or 0.0)
                cur_job_exp = normalize_expense_type(picked_row.get("job_expenses"))
                cur_exp_amt = float(picked_row.get("expenses_amount") or 0.0)
                cur_auth = str(picked_row.get("auth_code") or "").strip()
                cur_status = normalize_status(picked_row.get("job_status"))
                cur_waiting = str(picked_row.get("waiting_time") or "").strip()

                nonce = st.session_state.edit_nonce
                form_key = f"edit_form_{row_id}_{nonce}"

                with st.form(form_key):
                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        new_date = st.date_input("Date", value=cur_date, key=f"e_date_{row_id}_{nonce}")
                        new_job_number = st.text_input("job number (required)", value=cur_job_number, key=f"e_job_{row_id}_{nonce}")
                        new_job_type = st.selectbox("job type", JOB_TYPE_OPTIONS, index=JOB_TYPE_OPTIONS.index(cur_job_type), key=f"e_type_{row_id}_{nonce}")
                        new_status = st.selectbox("job status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(cur_status), key=f"e_status_{row_id}_{nonce}")

                    with c2:
                        new_vdesc = st.text_input("vehcile description", value=cur_vdesc, key=f"e_vdesc_{row_id}_{nonce}")
                        new_vreg = st.text_input("vehicle Reg", value=cur_vreg, key=f"e_vreg_{row_id}_{nonce}")

                    with c3:
                        new_from = st.text_input("collection from", value=cur_from, key=f"e_from_{row_id}_{nonce}")
                        new_to = st.text_input("delivery to", value=cur_to, key=f"e_to_{row_id}_{nonce}")

                    with c4:
                        new_job_amount = st.number_input("job amount", step=0.5, value=float(cur_job_amount), key=f"e_amt_{row_id}_{nonce}")
                        new_job_expenses = st.selectbox("Job Expenses", JOB_EXPENSE_OPTIONS, index=JOB_EXPENSE_OPTIONS.index(cur_job_exp), key=f"e_expt_{row_id}_{nonce}")
                        new_exp_amt = st.number_input("expenses Amount", step=0.5, value=float(cur_exp_amt), key=f"e_expa_{row_id}_{nonce}")
                        new_auth = st.text_input("Auth code", value=cur_auth, key=f"e_auth_{row_id}_{nonce}")

                        new_waiting_raw = st.text_input("waiting time", value=cur_waiting, key=f"e_wait_{row_id}_{nonce}")
                        wh, wn = parse_waiting_time(new_waiting_raw)
                        if new_waiting_raw.strip():
                            if wh is None:
                                st.error("Waiting time format invalid.")
                            else:
                                st.write(f"Waiting: **{wn}** | Hours: **{wh:.2f}** | Owed: **£{(wh*WAITING_RATE):.2f}**")

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
                            )
                            st.session_state.edit_selected_job = jn
                            st.session_state.edit_selected_row_id = row_id
                            st.session_state.edit_nonce += 1  # force refresh
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
            }
        )

        for col in UI_COLUMNS:
            if col not in view_df.columns:
                view_df[col] = ""

        st.dataframe(view_df[UI_COLUMNS], use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Weekly summary (deduped by Date+Job)")

        dfw = report_df.copy()
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
            weekly[c] = pd.to_numeric(weekly[c], errors="coerce").fillna(0)

        weekly["total_owed"] = weekly["job_amount"] + weekly["waiting_owed"]
        st.dataframe(weekly, use_container_width=True, hide_index=True)

        csv_bytes = view_df[UI_COLUMNS].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv_bytes,
            file_name="worklog_filtered.csv",
            mime="text/csv",
        )
