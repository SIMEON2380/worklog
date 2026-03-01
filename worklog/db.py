import os
import re
import sqlite3
from datetime import date, timedelta
from typing import Optional, Tuple, Any, Dict, List

import pandas as pd

from .config import Config


def make_db(cfg: Config):
    # closure pattern so you can pass cfg without globals
    CFG = cfg

    # =========================
    # Normalization helpers
    # =========================
    def normalize_status(x: Any) -> str:
        s = str(x or "").strip()
        return s if s in CFG.STATUS_OPTIONS else "Pending"

    def normalize_job_type(x: Any) -> str:
        s = str(x or "").strip()
        return s if s in CFG.JOB_TYPE_OPTIONS else CFG.JOB_TYPE_OPTIONS[0]

    def _expense_tokenize(raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, (list, tuple, set)):
            tokens = [str(x).strip() for x in raw]
            return [t for t in tokens if t]
        s = str(raw).strip()
        if not s or s.lower() == "nan":
            return []
        parts = re.split(r"[,\;\|]+", s)
        return [p.strip() for p in parts if p.strip()]

    def normalize_expense_types(raw: Any) -> str:
        tokens = _expense_tokenize(raw)
        allowed = set([o.lower().strip() for o in CFG.JOB_EXPENSE_OPTIONS])

        norm: List[str] = []
        for t in tokens:
            tl = t.lower().strip()
            if tl in allowed:
                norm.append(tl)
            else:
                norm.append("other")

        if not norm:
            norm = ["no expenses"]

        if "no expenses" in norm and len(set(norm)) > 1:
            norm = [x for x in norm if x != "no expenses"]

        seen = set()
        out = []
        for x in norm:
            if x not in seen:
                out.append(x)
                seen.add(x)

        return ", ".join(out)

    def clean_job_number(val: Any) -> str:
        if val is None:
            return ""
        s = str(val).strip()
        if s == "" or s.lower() == "nan":
            return ""
        s = s.replace(",", "")
        if re.fullmatch(r"\d+\.0+", s):
            s = s.split(".")[0]
        return s.strip()

    def coerce_money(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            f = float(x)
        except Exception:
            return None
        if f == 0.0:
            return None
        return float(f)

    # =========================
    # DB
    # =========================
    def ensure_db_dir():
        os.makedirs(CFG.DB_DIR, exist_ok=True)

    def get_conn() -> sqlite3.Connection:
        ensure_db_dir()
        conn = sqlite3.connect(CFG.DB_PATH, check_same_thread=False)
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
                CREATE TABLE IF NOT EXISTS {CFG.TABLE_NAME} (
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

            cols = table_columns(conn, CFG.TABLE_NAME)
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
                    conn.execute(f"ALTER TABLE {CFG.TABLE_NAME} ADD COLUMN {col} {col_type}")
            conn.commit()

            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{CFG.TABLE_NAME}_work_date ON {CFG.TABLE_NAME}(work_date)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{CFG.TABLE_NAME}_job_id ON {CFG.TABLE_NAME}(job_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{CFG.TABLE_NAME}_status ON {CFG.TABLE_NAME}(job_status)")
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
            df = pd.read_sql_query(f"SELECT * FROM {CFG.TABLE_NAME} ORDER BY work_date DESC, id DESC", conn)
        if df.empty:
            return pd.DataFrame(columns=CFG.EXPECTED_DB_COLS)

        for c in CFG.EXPECTED_DB_COLS:
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

        df["vehicle_description"] = df["vehicle_description"].fillna("").astype(str).str.strip().str.upper()
        df["vehicle_reg"] = df["vehicle_reg"].fillna("").astype(str)
        df["collection_from"] = df["collection_from"].fillna("").astype(str)
        df["delivery_to"] = df["delivery_to"].fillna("").astype(str)

        df["job_expenses"] = df["job_expenses"].fillna("").astype(str).apply(normalize_expense_types)

        df["auth_code"] = df["auth_code"].fillna("").astype(str)
        df["comments"] = df["comments"].fillna("").astype(str)

        return df

    def read_rows_by_job_number(job_number: str) -> pd.DataFrame:
        job_number = clean_job_number(job_number)
        if not job_number:
            return pd.DataFrame()

        with get_conn() as conn:
            df = pd.read_sql_query(
                f"""
                SELECT * FROM {CFG.TABLE_NAME}
                WHERE TRIM(COALESCE(job_id,'')) = ?
                ORDER BY work_date DESC, id DESC
                """,
                conn,
                params=(job_number,),
            )

        if df.empty:
            return df

        for c in CFG.EXPECTED_DB_COLS:
            if c not in df.columns:
                df[c] = None

        df["work_date"] = to_clean_date_series(df["work_date"])
        df["job_id"] = df["job_id"].apply(clean_job_number)
        df["vehicle_description"] = df["vehicle_description"].fillna("").astype(str).str.strip().str.upper()
        df["job_expenses"] = df["job_expenses"].fillna("").astype(str).apply(normalize_expense_types)
        df["comments"] = df["comments"].fillna("").astype(str)
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
        job_expenses: Any,
        expenses_amount: Optional[float],
        auth_code: str,
        job_status: str,
        waiting_time_raw: str,
        comments: str,
    ):
        job_number = clean_job_number(job_number)

        w_hours, w_norm = parse_waiting_time(waiting_time_raw)
        w_amount = (w_hours * CFG.WAITING_RATE) if (w_hours is not None) else None

        wd = work_date_val.isoformat()
        status = normalize_status(job_status)
        jt = normalize_job_type(job_type)
        je = normalize_expense_types(job_expenses)

        vdesc = str(vehicle_description or "").strip().upper()
        vreg = str(vehicle_reg or "").strip()
        cfrom = str(collection_from or "").strip()
        cto = str(delivery_to or "").strip()
        auth = str(auth_code or "").strip()
        cmts = str(comments or "").strip()

        amt = coerce_money(job_amount)
        exp_amt = coerce_money(expenses_amount)

        with get_conn() as conn:
            conn.execute(
                f"""
                INSERT INTO {CFG.TABLE_NAME}
                  (
                    work_date, job_id, category,
                    vehicle_description, vehicle_reg, collection_from, delivery_to,
                    amount, job_expenses, expenses_amount, auth_code, job_status,
                    waiting_time, waiting_hours, waiting_amount,
                    description, hours,
                    comments
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
                    amt,
                    je,
                    exp_amt,
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
        job_expenses: Any,
        expenses_amount: Optional[float],
        auth_code: str,
        job_status: str,
        waiting_time_raw: str,
        comments: str,
    ):
        job_number = clean_job_number(job_number)

        w_hours, w_norm = parse_waiting_time(waiting_time_raw)
        w_amount = (w_hours * CFG.WAITING_RATE) if (w_hours is not None) else None

        wd = work_date_val.isoformat()
        status = normalize_status(job_status)
        jt = normalize_job_type(job_type)
        je = normalize_expense_types(job_expenses)

        vdesc = str(vehicle_description or "").strip().upper()
        vreg = str(vehicle_reg or "").strip()
        cfrom = str(collection_from or "").strip()
        cto = str(delivery_to or "").strip()
        auth = str(auth_code or "").strip()
        cmts = str(comments or "").strip()

        amt = coerce_money(job_amount)
        exp_amt = coerce_money(expenses_amount)

        with get_conn() as conn:
            conn.execute(
                f"""
                UPDATE {CFG.TABLE_NAME}
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
                    waiting_amount = ?,
                    comments = ?
                WHERE id = ?
                """,
                (
                    wd,
                    job_number,
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
                    w_norm or "",
                    w_hours,
                    w_amount,
                    cmts,
                    int(row_id),
                ),
            )
            conn.commit()

    def delete_row_by_id(row_id: int) -> None:
        with get_conn() as conn:
            conn.execute(f"DELETE FROM {CFG.TABLE_NAME} WHERE id = ?", (int(row_id),))
            conn.commit()

    # Insert many, dedupe, reporting helpers stay in ui.py (so db.py doesn’t get even bigger).
    # But we still expose the key pieces used everywhere.
    return {
        "get_conn": get_conn,
        "ensure_schema": ensure_schema,
        "safe_date_bounds": safe_date_bounds,
        "to_clean_date_series": to_clean_date_series,
        "week_start": week_start,
        "parse_waiting_time": parse_waiting_time,
        "normalize_status": normalize_status,
        "normalize_job_type": normalize_job_type,
        "normalize_expense_types": normalize_expense_types,
        "clean_job_number": clean_job_number,
        "coerce_money": coerce_money,
        "read_all": read_all,
        "read_rows_by_job_number": read_rows_by_job_number,
        "insert_row": insert_row,
        "update_row_by_id": update_row_by_id,
        "delete_row_by_id": delete_row_by_id,
    }