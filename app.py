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
    Keeps legacy columns (hours/category) so old data stays intact.
    """
    with get_conn() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                work_date TEXT,
                description TEXT,
                hours REAL,            -- legacy (we hide it in UI)
                amount REAL,
                job_id TEXT,
                category TEXT,         -- legacy name (we use it as JOB TYPE)

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
    df["waiting_hours"] = pd.to_numeric(df.get("waiting_hours"), errors="coerce")
    df["waiting_amount"] = pd.to_numeric(df.get("waiting_amount"), errors="coerce")

    df["job_id"] = df.get("job_id", "").fillna("").astype(str)
    df["description"] = df.get("description", "").fillna("").astype(str)
    df["waiting_time"] = df.get("waiting_time", "").fillna("").astype(str)

    # legacy "category" treated as job_type in UI
    df["job_type"] = df.get("category", "").fillna("").astype(str)

    df["job_status"] = df.get("job_status", "").fillna("").astype(str)
    df["job_status"] = df["job_status"].apply(lambda x: x if x in STATUS_OPTIONS else "Pending")

    return df


def read_row_by_id(row_id: int) -> Optional[dict]:
    """Always read the row fresh from DB (prevents edit weirdness)."""
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(f"SELECT * FROM {TABLE_NAME} WHERE id = ?", (row_id,)).fetchone()
        return dict(row) if row else None


def insert_row(
    work_date_val: date,
    description: str,
    amount: Optional[float],
    job_id: str,
    job_type: str,
    job_status: str,
    waiting_time_raw: str,
):
    w_hours, w_norm = parse_waiting_time(waiting_time_raw)
    w_amount = (w_hours * WAITING_RATE) if (w_hours is not None) else None

    wd = work_date_val.isoformat()
    status = job_status if job_status in STATUS_OPTIONS else "Pending"
    jt = job_type if job_type in JOB_TYPE_OPTIONS else ""

    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME}
              (work_date, description, hours, amount, job_id, category, job_status, waiting_time, waiting_hours, waiting_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (wd, description, None, amount, job_id, jt, status, w_norm, w_hours, w_amount),
        )
        conn.commit()


def update_row(
    row_id: int,
    work_date_val: date,
    description: str,
    amount: Optional[float],
    job_id: str,
    job_type: str,
    job_status: str,
    waiting_time_raw: str,
):
    w_hours, w_norm = parse_waiting_time(waiting_time_raw)
    w_amount = (w_hours * WAITING_RATE) if (w_hours is not None) else None

    wd = work_date_val.isoformat()
    status = job_status if job_status in STATUS_OPTIONS else "Pending"
    jt = job_type if job_type in JOB_TYPE_OPTIONS else ""

    with get_conn() as conn:
        conn.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET
                work_date = ?,
                description = ?,
                hours = ?,
                amount = ?,
                job_id = ?,
                category = ?,
                job_status = ?,
                waiting_time = ?,
                waiting_hours = ?,
                waiting_amount = ?
            WHERE id = ?
            """,
            (wd, description, None, amount, job_id, jt, status, w_norm, w_hours, w_amount, row_id),
        )
        conn.commit()


def insert_many(df: pd.DataFrame) -> int:
    """
    Import Excel/CSV with flexible column names.
    Required: work_date/date and job_id/job_number
    Optional: description/comments, amount/job_amount, job_type/category/job_type, job_status/status, waiting_time
    """
    if df.empty:
        return 0

    df2 = df.copy()
    cols_lower = {c.lower().strip(): c for c in df2.columns}

    def pick(*names):
        for n in names:
            if n in cols_lower:
                return cols_lower[n]
        return None

    c_work_date = pick("work_date", "date")
    c_job_id = pick("job_id", "job_number", "job", "jobid")

    if c_work_date is None:
        raise ValueError("Upload must include work_date (or date).")
    if c_job_id is None:
        raise ValueError("Upload must include job_id (or job_number).")

    c_desc = pick("description", "comments", "desc", "details")
    c_amount = pick("amount", "job_amount", "value", "cost")
    c_job_type = pick("job_type", "job type", "category", "job_type", "job_type ")
    c_status = pick("job_status", "status")
    c_waiting = pick("waiting_time", "waiting", "waitingtime")

    df2["work_date"] = to_clean_date_series(df2[c_work_date])
    df2["job_id"] = df2[c_job_id].fillna("").astype(str).str.strip()
    df2["description"] = df2[c_desc].fillna("").astype(str) if c_desc else ""
    df2["amount"] = pd.to_numeric(df2[c_amount], errors="coerce") if c_amount else None

    df2["job_type"] = df2[c_job_type].fillna("").astype(str).str.strip() if c_job_type else ""
    df2["job_type"] = df2["job_type"].apply(lambda x: x if x in JOB_TYPE_OPTIONS else x)

    df2["job_status"] = df2[c_status].fillna("Pending").astype(str).str.strip() if c_status else "Pending"
    df2["job_status"] = df2["job_status"].apply(lambda x: x if x in STATUS_OPTIONS else "Pending")

    df2["waiting_raw"] = df2[c_waiting].fillna("").astype(str) if c_waiting else ""

    wh_list, wn_list, wa_list = [], [], []
    for txt in df2["waiting_raw"].tolist():
        wh, wn = parse_waiting_time(txt)
        wh_list.append(wh)
        wn_list.append(wn)
        wa_list.append((wh * WAITING_RATE) if wh is not None else None)

    df2["waiting_hours"] = wh_list
    df2["waiting_time"] = wn_list
    df2["waiting_amount"] = wa_list

    # enforce required fields
    df2 = df2[df2["work_date"].notna()].copy()
    df2 = df2[df2["job_id"] != ""].copy()
    if df2.empty:
        return 0

    df2["work_date"] = df2["work_date"].apply(lambda d: d.isoformat() if isinstance(d, date) else None)

    rows = list(
        zip(
            df2["work_date"].tolist(),
            df2["description"].tolist(),
            [None] * len(df2),  # hours hidden
            df2["amount"].tolist(),
            df2["job_id"].tolist(),
            df2["job_type"].tolist(),  # stored in legacy category
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
              (work_date, description, hours, amount, job_id, category, job_status, waiting_time, waiting_hours, waiting_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
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
                    COALESCE(description,''),
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
    search_txt = st.text_input("Search (description / job id / job type)", value="").strip()

df = df_all.copy()
if not df.empty:
    df = df[df["work_date"].notna()].copy()
    df = df[(df["work_date"] >= start_d) & (df["work_date"] <= end_d)]
    df = df[df["job_status"].isin(status_filter)]
    if search_txt:
        mask = (
            df["description"].str.contains(search_txt, case=False, na=False)
            | df["job_id"].astype(str).str.contains(search_txt, case=False, na=False)
            | df["job_type"].astype(str).str.contains(search_txt, case=False, na=False)
        )
        df = df[mask]

tab1, tab2, tab3 = st.tabs(["Add entry", "Upload Excel/CSV", "View & Edit"])

# -------- Add entry --------
with tab1:
    st.subheader("Add entry")

    c1, c2, c3 = st.columns(3)
    with c1:
        work_date_val = st.date_input("Work date", value=date.today())
        job_status = st.selectbox("Job status", STATUS_OPTIONS, index=STATUS_OPTIONS.index("Pending"))
        job_id = st.text_input("Job ID (required)")
    with c2:
        job_type = st.selectbox("Job Type", JOB_TYPE_OPTIONS, index=0)
        amount = st.number_input("Amount (£) (job)", step=0.5, value=0.0)
    with c3:
        waiting_time_raw = st.text_input("Waiting time (e.g. 10-12 or 10:30-12:15)", value="")
        w_hours, w_norm = parse_waiting_time(waiting_time_raw)
        if waiting_time_raw.strip():
            if w_hours is None:
                st.error("Waiting time format invalid. Use like 10-12 or 10:30-12:15.")
            else:
                st.write(f"Waiting: **{w_norm}** | Hours: **{w_hours:.2f}** | Owed: **£{(w_hours*WAITING_RATE):.2f}**")

    description = st.text_area("Description", height=60)

    if st.button("Save entry"):
        if str(job_id).strip() == "":
            st.error("Job ID is required.")
        elif waiting_time_raw.strip() and w_hours is None:
            st.error("Fix waiting time format before saving.")
        else:
            insert_row(
                work_date_val=work_date_val,
                description=description,
                amount=amount,
                job_id=str(job_id).strip(),
                job_type=job_type,
                job_status=job_status,
                waiting_time_raw=waiting_time_raw,
            )
            st.success("Saved.")
            st.rerun()

# -------- Upload --------
with tab2:
    st.subheader("Upload Excel/CSV")
    st.write("Required: `work_date`/`date` and `job_id`/`job_number`. Optional: `amount/job_amount`, `comments/description`, `job_type/category`, `status`, `waiting_time`.")

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
        total_wait = df["waiting_amount"].fillna(0).sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", f"{len(df):,}")
        k2.metric("Job amount (£)", f"{total_job:,.2f}")
        k3.metric("Waiting owed (£)", f"{total_wait:,.2f}")
        k4.metric("Total owed (£)", f"{(total_job + total_wait):,.2f}")

        st.divider()
        st.markdown("## Edit (select an ID and save)")
        st.caption("The table below is view-only. Use the edit form here to update rows in the database.")

        id_list = df["id"].tolist()
        selected_id = st.selectbox("Select ID", options=id_list)

        db_row = read_row_by_id(int(selected_id))
        if not db_row:
            st.error("Could not load that row from the database.")
        else:
            cur_date_series = to_clean_date_series(pd.Series([db_row.get("work_date")]))
            cur_date = cur_date_series.iloc[0] if isinstance(cur_date_series.iloc[0], date) else date.today()

            cur_status = (db_row.get("job_status") or "Pending").strip()
            if cur_status not in STATUS_OPTIONS:
                cur_status = "Pending"

            cur_job_id = str(db_row.get("job_id") or "").strip()

            cur_job_type = str(db_row.get("category") or "").strip()
            if cur_job_type not in JOB_TYPE_OPTIONS:
                cur_job_type = JOB_TYPE_OPTIONS[0]

            cur_amount = db_row.get("amount")
            cur_amount = float(cur_amount) if cur_amount is not None else 0.0

            cur_waiting = str(db_row.get("waiting_time") or "").strip()
            cur_desc = str(db_row.get("description") or "")

            with st.form("edit_form"):
                c1, c2, c3 = st.columns(3)

                with c1:
                    new_date = st.date_input("Work date", value=cur_date)
                    new_status = st.selectbox("Job status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(cur_status))
                    new_job_id = st.text_input("Job ID (required)", value=cur_job_id)

                with c2:
                    new_job_type = st.selectbox("Job Type", JOB_TYPE_OPTIONS, index=JOB_TYPE_OPTIONS.index(cur_job_type))
                    new_amount = st.number_input("Amount (£) (job)", step=0.5, value=cur_amount)

                with c3:
                    new_waiting_raw = st.text_input("Waiting time", value=cur_waiting)
                    wh, wn = parse_waiting_time(new_waiting_raw)
                    if new_waiting_raw.strip():
                        if wh is None:
                            st.error("Waiting time format invalid. Use like 10-12 or 10:30-12:15.")
                        else:
                            st.write(f"Waiting: **{wn}** | Hours: **{wh:.2f}** | Owed: **£{(wh*WAITING_RATE):.2f}**")

                new_desc = st.text_area("Description", value=cur_desc, height=60)

                save = st.form_submit_button("Save changes")

                if save:
                    if str(new_job_id).strip() == "":
                        st.error("Job ID is required.")
                    elif new_waiting_raw.strip() and wh is None:
                        st.error("Fix waiting time format before saving.")
                    else:
                        update_row(
                            row_id=int(selected_id),
                            work_date_val=new_date,
                            description=new_desc,
                            amount=float(new_amount),
                            job_id=str(new_job_id).strip(),
                            job_type=new_job_type,
                            job_status=new_status,
                            waiting_time_raw=new_waiting_raw,
                        )
                        st.success("Updated.")
                        st.rerun()

        st.divider()
        st.subheader("Records (view only)")
        st.dataframe(
            df[
                [
                    "id",
                    "work_date",
                    "job_id",
                    "job_status",
                    "job_type",
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
                job_amount=("amount", "sum"),
                waiting_hours=("waiting_hours", "sum"),
                waiting_owed=("waiting_amount", "sum"),
            )
            .sort_values("week_start", ascending=False)
        )
        for c in ["job_amount", "waiting_hours", "waiting_owed"]:
            weekly[c] = weekly[c].fillna(0)
        weekly["total_owed"] = weekly["job_amount"] + weekly["waiting_owed"]
        st.dataframe(weekly, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", data=csv_bytes, file_name="worklog_filtered.csv", mime="text/csv")
