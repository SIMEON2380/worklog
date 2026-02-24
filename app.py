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

    # Vehicle description ALWAYS uppercase (including old rows display)
    df["vehicle_description"] = df["vehicle_description"].fillna("").astype(str).str.strip().str.upper()
    df["vehicle_reg"] = df["vehicle_reg"].fillna("").astype(str)
    df["collection_from"] = df["collection_from"].fillna("").astype(str)
    df["delivery_to"] = df["delivery_to"].fillna("").astype(str)
    df["job_expenses"] = df["job_expenses"].fillna("").astype(str).apply(normalize_expense_type)
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
    df["vehicle_description"] = df["vehicle_description"].fillna("").astype(str).str.strip().str.upper()
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
    job_expenses: str,
    expenses_amount: Optional[float],
    auth_code: str,
    job_status: str,
    waiting_time_raw: str,
    comments: str,
):
    job_number = clean_job_number(job_number)

    w_hours, w_norm = parse_waiting_time(waiting_time_raw)
    w_amount = (w_hours * WAITING_RATE) if (w_hours is not None) else None

    wd = work_date_val.isoformat()
    status = normalize_status(job_status)
    jt = normalize_job_type(job_type)
    je = normalize_expense_type(job_expenses)

    vdesc = str(vehicle_description or "").strip().upper()  # ALWAYS CAPS
    vreg = str(vehicle_reg or "").strip()
    cfrom = str(collection_from or "").strip()
    cto = str(delivery_to or "").strip()
    auth = str(auth_code or "").strip()
    cmts = str(comments or "").strip()

    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME}
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
                float(job_amount) if job_amount is not None else None,
                je,
                float(expenses_amount) if expenses_amount is not None else None,
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
    job_expenses: str,
    expenses_amount: Optional[float],
    auth_code: str,
    job_status: str,
    waiting_time_raw: str,
    comments: str,
):
    job_number = clean_job_number(job_number)

    w_hours, w_norm = parse_waiting_time(waiting_time_raw)
    w_amount = (w_hours * WAITING_RATE) if (w_hours is not None) else None

    wd = work_date_val.isoformat()
    status = normalize_status(job_status)
    jt = normalize_job_type(job_type)
    je = normalize_expense_type(job_expenses)

    vdesc = str(vehicle_description or "").strip().upper()  # ALWAYS CAPS
    vreg = str(vehicle_reg or "").strip()
    cfrom = str(collection_from or "").strip()
    cto = str(delivery_to or "").strip()
    auth = str(auth_code or "").strip()
    cmts = str(comments or "").strip()

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
                float(job_amount) if job_amount is not None else None,
                je,
                float(expenses_amount) if expenses_amount is not None else None,
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
    c_comments = pick("comments", "comment", "notes", "note")

    df2["work_date"] = to_clean_date_series(df2[c_date])
    df2["job_id"] = df2[c_job].apply(clean_job_number)

    df2["job_type"] = df2[c_job_type].fillna("").astype(str).str.strip() if c_job_type else JOB_TYPE_OPTIONS[0]
    df2["job_type"] = df2["job_type"].apply(normalize_job_type)

    # vehicle description ALWAYS uppercase
    df2["vehicle_description"] = (
        df2[c_vdesc].fillna("").astype(str).str.strip().str.upper()
        if c_vdesc else ""
    )
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

    df2["comments"] = df2[c_comments].fillna("").astype(str).str.strip() if c_comments else ""

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
            df2["comments"].tolist(),
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
                description, hours,
                comments
              )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    cmts,
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
                    cmts,
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

        # ---- FIX: weekly summary should never KeyError if columns differ ----
        def pick_existing_col(dff: pd.DataFrame, *candidates: str) -> Optional[str]:
            """Return the first candidate that exists in dff.columns, else None."""
            for c in candidates:
                if c in dff.columns:
                    return c
            return None

        dfw = report_df.copy()

        # Exclude Withdraw rows consistently
        status_col = pick_existing_col(dfw, "job_status", "job status")
        if status_col is not None:
            dfw = dfw[dfw[status_col].astype(str).str.lower().str.strip() != "withdraw"].copy()

        # Date column
        date_col = pick_existing_col(dfw, "work_date", "Date")
        if date_col is None:
            st.error("Weekly summary error: no date column found (expected 'work_date' or 'Date').")
            st.stop()

        # Normalize dates -> week_start
        dfw["_work_date_for_week"] = to_clean_date_series(dfw[date_col])
        dfw = dfw[dfw["_work_date_for_week"].notna()].copy()
        dfw["week_start"] = dfw["_work_date_for_week"].apply(week_start)

        # Money columns (accept DB names + UI names + a few common variants)
        amount_col = pick_existing_col(dfw, "amount", "job amount", "job_amount", "Job amount", "Job Amount")
        exp_col = pick_existing_col(dfw, "expenses_amount", "expenses Amount", "expenses amount", "Expenses Amount")
        wait_col = pick_existing_col(
            dfw,
            "waiting_amount",
            "waiting owed",
            "waiting_owed",
            "waiting amount",
            "waiting_owed_for_week",
            "waiting owed for week",
        )

        # Create missing columns as zeros so groupby never crashes
        if amount_col is None:
            dfw["_job_amount_for_week"] = 0.0
            amount_col = "_job_amount_for_week"
        if exp_col is None:
            dfw["_expenses_amount_for_week"] = 0.0
            exp_col = "_expenses_amount_for_week"
        if wait_col is None:
            dfw["_waiting_owed_for_week"] = 0.0
            wait_col = "_waiting_owed_for_week"

        # Coerce numeric safely
        dfw[amount_col] = pd.to_numeric(dfw[amount_col], errors="coerce").fillna(0)
        dfw[exp_col] = pd.to_numeric(dfw[exp_col], errors="coerce").fillna(0)
        dfw[wait_col] = pd.to_numeric(dfw[wait_col], errors="coerce").fillna(0)

        # Row count: prefer unique jobs per week
        job_id_col = pick_existing_col(dfw, "job_id", "job number", "job_number", "job id")
        if job_id_col is not None:
            dfw["_job_key_for_week"] = dfw[job_id_col].astype(str).str.strip()
            rows_agg = ("_job_key_for_week", "nunique")
        elif "id" in dfw.columns:
            rows_agg = ("id", "count")
        else:
            dfw["_row_counter"] = 1
            rows_agg = ("_row_counter", "sum")

        weekly = (
            dfw.groupby("week_start", as_index=False)
            .agg(
                rows=rows_agg,
                job_amount=(amount_col, "sum"),
                expenses_amount=(exp_col, "sum"),
                waiting_owed=(wait_col, "sum"),
            )
            .sort_values("week_start", ascending=False)
        )

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
