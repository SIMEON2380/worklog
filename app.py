import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user, verify_login, change_password

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=cfg.APP_TITLE, layout="wide")

# Init DB + users
DB["ensure_schema"]()
ensure_default_user(cfg)

# Session state
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None


# ---------- helpers ----------
def _pick_date_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "job_date",
        "date",
        "work_date",
        "created_at",
        "updated_at",
        "timestamp",
        "time",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def get_jobs_df() -> pd.DataFrame:
    """
    Tries common DB function keys. Adjust only if your DB uses a different key.
    """
    candidates = [
        "get_jobs_df",
        "get_all_jobs_df",
        "fetch_all_df",
        "read_all_df",
        "all_jobs_df",
        "select_all_df",
    ]

    for key in candidates:
        fn = DB.get(key)
        if callable(fn):
            df = fn()
            if df is None:
                return pd.DataFrame()
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            return df

    fallback_candidates = ["get_jobs", "fetch_all", "read_all", "select_all"]
    for key in fallback_candidates:
        fn = DB.get(key)
        if callable(fn):
            rows = fn()
            if rows is None:
                return pd.DataFrame()
            return pd.DataFrame(rows)

    st.error(
        "Can't find a DB function to load jobs as a DataFrame.\n\n"
        "Add one of these keys in worklog/db.py make_db(): "
        "get_jobs_df / get_all_jobs_df / fetch_all_df / read_all_df."
    )
    st.stop()


def _find_status_col(df: pd.DataFrame) -> str | None:
    for c in ["job_status", "status", "Job Status", "Status"]:
        if c in df.columns:
            return c
    return None


def _find_money_col(df: pd.DataFrame) -> str | None:
    for c in ["total", "amount", "price", "pay", "earnings", "total_pay"]:
        if c in df.columns:
            return c
    return None


def render_report(df: pd.DataFrame, mode: str):
    """
    mode: "daily" | "weekly" | "monthly"
    """
    title = {"daily": "Daily Report", "weekly": "Weekly Report", "monthly": "Monthly Report"}[mode]
    st.subheader(title)

    if df.empty:
        st.info("No jobs found.")
        return

    date_col = _pick_date_col(df)
    if not date_col:
        st.warning(
            "Report needs a date column (job_date/date/created_at/etc). "
            "I couldn't find one in your table."
        )
        st.write("Columns found:", list(df.columns))
        return

    df = df.copy()
    df[date_col] = _coerce_datetime(df[date_col])
    df = df.dropna(subset=[date_col])
    if df.empty:
        st.info("No rows have a valid date, so reports can't calculate.")
        return

    # Build grouping key
    df["_day"] = df[date_col].dt.date

    if mode == "daily":
        df["_period"] = df["_day"]

        label = "Select day"
        periods = sorted(df["_period"].unique(), reverse=True)

    elif mode == "weekly":
        # Monday-start week
        day_dt = pd.to_datetime(df["_day"])
        week_start = (day_dt - pd.to_timedelta(day_dt.dt.dayofweek, unit="D")).dt.date
        df["_period"] = week_start

        label = "Select week (Mon–Sun)"
        periods = sorted(df["_period"].unique(), reverse=True)

    elif mode == "monthly":
        # YYYY-MM
        df["_period"] = df[date_col].dt.to_period("M").astype(str)

        label = "Select month"
        periods = sorted(df["_period"].unique(), reverse=True)

    else:
        st.error("Invalid report mode.")
        return

    selected = st.selectbox(label, periods, index=0)
    sub = df[df["_period"] == selected].copy()

    status_col = _find_status_col(sub)
    money_col = _find_money_col(sub)

    # Metrics
    total_jobs = len(sub)

    paid_jobs = 0
    if status_col:
        paid_jobs = (sub[status_col].astype(str).str.lower() == "paid").sum()

    total_money = None
    if money_col:
        sub[money_col] = pd.to_numeric(sub[money_col], errors="coerce")
        total_money = float(sub[money_col].fillna(0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Jobs", int(total_jobs))
    c2.metric("Paid jobs", int(paid_jobs))
    c3.metric("Total", f"£{total_money:,.2f}" if total_money is not None else "—")

    # Status breakdown
    if status_col:
        st.caption("Status breakdown")
        vc = (
            sub[status_col]
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("status")
            .reset_index(name="count")
        )
        st.dataframe(vc, use_container_width=True)

    # Rows
    st.caption("Jobs in selected period")
    st.dataframe(
        sub.drop(columns=["_day", "_period"], errors="ignore"),
        use_container_width=True,
    )


# ---------- Auth UI ----------
def render_login():
    st.title(cfg.APP_TITLE)
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = verify_login(DB, username, password)
        if user:
            st.session_state.auth_user = user
            st.success("Logged in")
            st.rerun()
        else:
            st.error("Invalid credentials")


def render_account():
    st.subheader("Account")
    st.caption(f"Signed in as: {st.session_state.auth_user}")

    with st.expander("Change password", expanded=False):
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        new2 = st.text_input("Confirm new password", type="password")

        if st.button("Update password"):
            if new != new2:
                st.error("New passwords do not match.")
            else:
                ok, msg = change_password(DB, st.session_state.auth_user, current, new)
                if ok:
                    st.success("Password changed.")
                else:
                    st.error(msg)

    if st.button("Logout"):
        st.session_state.auth_user = None
        st.rerun()


# ---------- Main ----------
if not st.session_state.auth_user:
    render_login()
    st.stop()

df = get_jobs_df()

st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Daily Report",
        "Weekly Report",
        "Monthly Report",
        "Account",
    ],
)

if page == "Dashboard":
    st.title("Dashboard")
    st.caption("All jobs in the database")
    st.dataframe(df, use_container_width=True)

elif page == "Daily Report":
    render_report(df, "daily")

elif page == "Weekly Report":
    render_report(df, "weekly")

elif page == "Monthly Report":
    render_report(df, "monthly")

elif page == "Account":
    render_account()