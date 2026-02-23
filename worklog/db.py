import os
import sqlite3
from typing import Iterable, Optional, Dict, Any, Tuple
import pandas as pd

from worklog.normalize import clean_job_number, normalize_status, normalize_job_type, normalize_expense_type

TABLE_NAME = "work_logs"


def get_db_path() -> str:
    db_dir = os.environ.get("WORKLOG_DB_DIR", "/var/lib/worklog")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "worklog.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
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

        # unique protection for future duplicates
        try:
            conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{TABLE_NAME}_date_job ON {TABLE_NAME}(work_date, job_id)"
            )
            conn.commit()
        except Exception:
            pass


def read_all() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY work_date DESC, id DESC", conn)


def update_status_for_year(year: int, new_status: str = "Paid") -> int:
    """
    Set all rows with work_date in a given year to new_status.
    """
    ns = normalize_status(new_status)
    y = str(year)
    with get_conn() as conn:
        cur = conn.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET job_status = ?
            WHERE substr(COALESCE(work_date,''), 1, 4) = ?
            """,
            (ns, y),
        )
        conn.commit()
        return int(cur.rowcount or 0)


def backfill_from_dataframe(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Backfill missing fields by matching (work_date, job_id).
    Updates ONLY when DB value is empty/NULL.
    Returns (matched_rows, updated_rows).
    """
    if df is None or df.empty:
        return 0, 0

    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            k = str(n).lower().strip()
            if k in cols_lower:
                return cols_lower[k]
        return None

    c_date = pick("date", "work_date")
    c_job = pick("job number", "job_number", "job_id", "jobid", "job")
    if c_date is None or c_job is None:
        raise ValueError("Backfill file must include Date and job number columns.")

    c_vdesc = pick("vehcile description", "vehicle description", "vehicle_description")
    c_vreg = pick("vehicle reg", "vehicle_reg", "vehicle Reg")
    c_from = pick("collection from", "collection_from")
    c_to = pick("delivery to", "delivery_to")
    c_auth = pick("auth code", "auth_code", "Auth code")
    c_status = pick("job status", "job_status", "status")
    c_comments = pick("comments", "comment", "notes", "note")
    c_job_type = pick("job type", "job_type", "category")
    c_exp_type = pick("job expenses", "job_expenses", "Job Expenses")
    c_amt = pick("job amount", "amount", "job_amount")
    c_exp_amt = pick("expenses amount", "expenses_amount", "expenses Amount")
    c_waiting = pick("waiting time", "waiting_time", "waiting")

    # normalize inputs
    d = df.copy()
    d["work_date"] = pd.to_datetime(d[c_date], errors="coerce").dt.date
    d["job_id"] = d[c_job].apply(clean_job_number)
    d = d[d["work_date"].notna() & (d["job_id"] != "")].copy()
    if d.empty:
        return 0, 0

    def get_str(colname, upper=False):
        if colname is None:
            return ""
        s = d[colname].fillna("").astype(str).str.strip()
        return s.str.upper() if upper else s

    d["vehicle_description"] = get_str(c_vdesc, upper=True)
    d["vehicle_reg"] = get_str(c_vreg)
    d["collection_from"] = get_str(c_from)
    d["delivery_to"] = get_str(c_to)
    d["auth_code"] = get_str(c_auth)
    d["comments"] = get_str(c_comments)

    if c_status:
        d["job_status"] = d[c_status].fillna("").astype(str).str.strip().apply(normalize_status)
    else:
        d["job_status"] = ""

    if c_job_type:
        d["category"] = d[c_job_type].fillna("").astype(str).str.strip().apply(normalize_job_type)
    else:
        d["category"] = ""

    if c_exp_type:
        d["job_expenses"] = d[c_exp_type].fillna("").astype(str).str.strip().apply(normalize_expense_type)
    else:
        d["job_expenses"] = ""

    # numeric fields optional
    d["amount"] = pd.to_numeric(d[c_amt], errors="coerce") if c_amt else None
    d["expenses_amount"] = pd.to_numeric(d[c_exp_amt], errors="coerce") if c_exp_amt else None
    d["waiting_time"] = d[c_waiting].fillna("").astype(str).str.strip() if c_waiting else ""

    matched = 0
    updated = 0

    with get_conn() as conn:
        for _, r in d.iterrows():
            wd = r["work_date"].isoformat()
            jid = r["job_id"]

            row = conn.execute(
                f"""
                SELECT id,
                       COALESCE(vehicle_description,''), COALESCE(vehicle_reg,''),
                       COALESCE(collection_from,''), COALESCE(delivery_to,''),
                       COALESCE(auth_code,''), COALESCE(job_status,''),
                       COALESCE(category,''), COALESCE(job_expenses,''),
                       COALESCE(comments,'')
                FROM {TABLE_NAME}
                WHERE work_date = ? AND job_id = ?
                """,
                (wd, jid),
            ).fetchone()

            if not row:
                continue

            matched += 1
            row_id = int(row[0])

            def is_empty(s: str) -> bool:
                return str(s or "").strip() == ""

            sets: Dict[str, Any] = {}

            if is_empty(row[1]) and str(r["vehicle_description"]).strip():
                sets["vehicle_description"] = str(r["vehicle_description"]).strip().upper()
            if is_empty(row[2]) and str(r["vehicle_reg"]).strip():
                sets["vehicle_reg"] = str(r["vehicle_reg"]).strip()
            if is_empty(row[3]) and str(r["collection_from"]).strip():
                sets["collection_from"] = str(r["collection_from"]).strip()
            if is_empty(row[4]) and str(r["delivery_to"]).strip():
                sets["delivery_to"] = str(r["delivery_to"]).strip()
            if is_empty(row[5]) and str(r["auth_code"]).strip():
                sets["auth_code"] = str(r["auth_code"]).strip()
            if is_empty(row[9]) and str(r["comments"]).strip():
                sets["comments"] = str(r["comments"]).strip()

            # Optional: only backfill status if DB empty
            if is_empty(row[6]) and str(r["job_status"]).strip():
                sets["job_status"] = str(r["job_status"]).strip()

            # Optional: only backfill job type if DB empty
            if is_empty(row[7]) and str(r["category"]).strip():
                sets["category"] = str(r["category"]).strip()

            # Optional: only backfill expense type if DB empty
            if is_empty(row[8]) and str(r["job_expenses"]).strip():
                sets["job_expenses"] = str(r["job_expenses"]).strip()

            if not sets:
                continue

            cols = ", ".join([f"{k} = ?" for k in sets.keys()])
            vals = list(sets.values()) + [row_id]

            conn.execute(f"UPDATE {TABLE_NAME} SET {cols} WHERE id = ?", vals)
            updated += 1

        conn.commit()

    return matched, updated