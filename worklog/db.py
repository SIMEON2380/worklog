import os
import re
import sqlite3
from typing import Dict, Any, Tuple, List

import pandas as pd

from worklog.normalize import (
    clean_job_number,
    normalize_status,
    normalize_job_type,
    normalize_expense_type,
)

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

    Default:
      - Only fills fields that are empty in DB.

    Upgrade behavior:
      - For collection_from / delivery_to:
        if DB value has NO postcode but incoming value DOES, update (upgrade) it.

    Robustness (IMPORTANT):
      - Some sheets have the job number in different columns on different rows
        (e.g. Job Number contains 'Std Trade Plate' and the real number is in Job type).
      - We therefore pick job_id PER ROW by trying multiple candidate columns.
    """
    if df is None or df.empty:
        return 0, 0

    # Practical UK postcode matcher (accepts no-space postcodes too)
    UK_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b", re.I)

    def has_postcode(s: str) -> bool:
        return bool(UK_POSTCODE_RE.search(str(s or "").upper()))

    # Map normalized column names -> real column names
    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            k = str(n).lower().strip()
            if k in cols_lower:
                return cols_lower[k]
        return None

    # Required-ish: Date
    c_date = pick("date", "work_date")
    if c_date is None:
        raise ValueError("Backfill file must include a Date/work_date column.")

    # Candidates for job id (we'll try multiple PER ROW)
    c_job = pick("job number", "job_number", "job_id", "jobid", "job no", "job")
    c_job_type = pick("job type", "job_type", "category", "Job type")

    # Also include any column that smells like it might hold an ID
    def guess_jobid_candidates(columns: List[str]) -> List[str]:
        out = []
        for c in columns:
            cl = str(c).lower()
            if "job" in cl or "order" in cl:
                out.append(c)
        return out

    guessed = guess_jobid_candidates(list(df.columns))

    # Ordered, de-duped candidate list
    jobid_candidates = []
    for c in [c_job, c_job_type, *guessed]:
        if c and c not in jobid_candidates:
            jobid_candidates.append(c)

    if not jobid_candidates:
        raise ValueError("Backfill file must include a job number column (Job Number / job_number / job_id).")

    # Other fields (support BCA-style headers too)
    c_vdesc = pick("vehcile description", "vehicle description", "vehicle_description", "Vehicle description", "job")
    c_vreg = pick("vehicle reg", "vehicle_reg", "vehicle Reg", "Vehicle Reg", "reg", "Reg", "registration")

    c_from = pick("collection from", "collection_from", "from", "From", "from_loc", "pickup", "pickup_from")
    c_to = pick("delivery to", "delivery_to", "to", "To", "to_loc", "dropoff", "dropoff_to")

    c_auth = pick("auth code", "auth_code", "Auth code", "AUTH CODE")
    c_status = pick("job status", "job_status", "status", "Job Status")
    c_comments = pick("comments", "comment", "notes", "note", "Comments")

    c_cat = pick("job type", "job_type", "category", "Job type")
    c_exp_type = pick("job expenses", "job_expenses", "Job Expenses", "Expense type")

    c_amt = pick("job amount", "amount", "job_amount", "Job Amount", "Amount")
    c_exp_amt = pick("expenses amount", "expenses_amount", "expenses Amount", "Expenses", "Expense Amount")
    c_waiting = pick("waiting time", "waiting_time", "waiting", "claimed", "Claimed")

    d = df.copy()

    # Date normalize
    d["work_date"] = pd.to_datetime(d[c_date], errors="coerce").dt.date
    d = d[d["work_date"].notna()].copy()
    if d.empty:
        return 0, 0

    # Build job_id PER ROW (this is the key fix)
    def pick_job_id_for_row(row) -> str:
        for col in jobid_candidates:
            try:
                v = clean_job_number(row.get(col, ""))
            except Exception:
                v = ""
            if str(v).strip():
                return str(v).strip()
        return ""

    d["job_id"] = d.apply(pick_job_id_for_row, axis=1)
    d = d[d["job_id"].astype(str).str.strip() != ""].copy()
    if d.empty:
        return 0, 0

    def get_str(colname, upper=False):
        if colname is None or colname not in d.columns:
            return ""
        s = d[colname].fillna("").astype(str).str.strip()
        return s.str.upper() if upper else s

    d["vehicle_description"] = get_str(c_vdesc, upper=True)
    d["vehicle_reg"] = get_str(c_vreg)
    d["collection_from"] = get_str(c_from)
    d["delivery_to"] = get_str(c_to)
    d["auth_code"] = get_str(c_auth)
    d["comments"] = get_str(c_comments)

    if c_status and c_status in d.columns:
        d["job_status"] = d[c_status].fillna("").astype(str).str.strip().apply(normalize_status)
    else:
        d["job_status"] = ""

    if c_cat and c_cat in d.columns:
        d["category"] = d[c_cat].fillna("").astype(str).str.strip().apply(normalize_job_type)
    else:
        d["category"] = ""

    if c_exp_type and c_exp_type in d.columns:
        d["job_expenses"] = d[c_exp_type].fillna("").astype(str).str.strip().apply(normalize_expense_type)
    else:
        d["job_expenses"] = ""

    d["amount"] = pd.to_numeric(d[c_amt], errors="coerce") if (c_amt and c_amt in d.columns) else None
    d["expenses_amount"] = pd.to_numeric(d[c_exp_amt], errors="coerce") if (c_exp_amt and c_exp_amt in d.columns) else None
    d["waiting_time"] = d[c_waiting].fillna("").astype(str).str.strip() if (c_waiting and c_waiting in d.columns) else ""

    matched = 0
    updated = 0

    with get_conn() as conn:
        for _, r in d.iterrows():
            wd = r["work_date"].isoformat()
            jid = str(r["job_id"]).strip()

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

            # Fill if empty
            if is_empty(row[1]) and str(r["vehicle_description"]).strip():
                sets["vehicle_description"] = str(r["vehicle_description"]).strip().upper()

            if is_empty(row[2]) and str(r["vehicle_reg"]).strip():
                sets["vehicle_reg"] = str(r["vehicle_reg"]).strip()

            # Fill if empty OR upgrade postcode
            old_from = str(row[3] or "").strip()
            new_from = str(r["collection_from"] or "").strip()
            if new_from and (is_empty(old_from) or (not has_postcode(old_from) and has_postcode(new_from))):
                sets["collection_from"] = new_from

            old_to = str(row[4] or "").strip()
            new_to = str(r["delivery_to"] or "").strip()
            if new_to and (is_empty(old_to) or (not has_postcode(old_to) and has_postcode(new_to))):
                sets["delivery_to"] = new_to

            if is_empty(row[5]) and str(r["auth_code"]).strip():
                sets["auth_code"] = str(r["auth_code"]).strip()

            if is_empty(row[9]) and str(r["comments"]).strip():
                sets["comments"] = str(r["comments"]).strip()

            # Optional fills (only if empty in DB)
            if is_empty(row[6]) and str(r["job_status"]).strip():
                sets["job_status"] = str(r["job_status"]).strip()

            if is_empty(row[7]) and str(r["category"]).strip():
                sets["category"] = str(r["category"]).strip()

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