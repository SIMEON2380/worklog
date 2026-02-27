import streamlit as st

from worklog.config import Config
from worklog.db import init_db, fetch_jobs
from worklog.auth import ensure_default_user, verify_login, change_password
from worklog.ui import (
    show_top_metrics,
    status_bar,
    job_entry_form,
    editable_jobs_table,
)

cfg = Config()

st.set_page_config(page_title=cfg.APP_TITLE, layout="wide")

# Init DB + users
init_db(cfg)
ensure_default_user(cfg)

# Session state
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# ---------- Auth ----------
def login_view():
    st.title(cfg.APP_TITLE)
    st.caption("Login required.")

    col1, col2 = st.columns([1, 2])
    with col1:
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")

        if st.button("Login", type="primary"):
            if verify_login(cfg, username, password):
                st.session_state.auth_user = username
                st.success("Logged in.")
                st.rerun()
            else:
                st.error("Wrong username or password.")


def sidebar_controls():
    with st.sidebar:
        st.write(f"👤 **{st.session_state.auth_user}**")
        page = st.radio(
            "Pages",
            ["Dashboard", "Inspect & Collect", "Status Board", "Change Password"],
            index=0,
        )

        if st.button("Logout"):
            st.session_state.auth_user = None
            st.rerun()

    return page


# ---------- Pages ----------
def page_dashboard():
    st.title("Dashboard")
    st.caption("All jobs in the database.")

    # Filters
    c1, c2, c3, c4 = st.columns(4)
    date_from = c1.date_input("From", value=None)
    date_to = c2.date_input("To", value=None)
    status = c3.selectbox("Status", ["All"] + cfg.STATUS_OPTIONS, index=0)
    job_type = c4.selectbox("Job type", ["All"] + cfg.JOB_TYPE_OPTIONS, index=0)
    search = st.text_input("Search (notes/reference)", placeholder="type anything...")

    df = fetch_jobs(
        cfg,
        date_from=str(date_from) if date_from else None,
        date_to=str(date_to) if date_to else None,
        status=status,
        job_type=job_type,
        search=search.strip() if search else None,
    )

    show_top_metrics(df)
    status_bar(df, cfg)

    job_entry_form(cfg)

    st.subheader("All jobs")
    editable_jobs_table(cfg, df, key="dashboard_editor", allow_type_edit=True)


def page_inspect_collect():
    st.title("Inspect & Collect")
    st.caption("Only Inspect & Collect jobs. You can edit these and set status to Paid.")

    df = fetch_jobs(cfg, job_type="All")
    if not df.empty:
        df = df[df["job_type"].isin(list(cfg.INSPECT_COLLECT_TYPES))].copy()

    show_top_metrics(df)
    st.subheader("Inspect & Collect jobs")
    editable_jobs_table(cfg, df, key="inspect_editor", allow_type_edit=False)


def page_status_board():
    st.title("Status Board")
    st.caption("View jobs by status and change job statuses.")

    picked = st.selectbox("Choose a status to view", cfg.STATUS_OPTIONS, index=0)
    df = fetch_jobs(cfg, status=picked)

    show_top_metrics(df)

    st.subheader(f"Jobs with status: {picked}")
    editable_jobs_table(cfg, df, key="status_editor", allow_type_edit=True)


def page_change_password():
    st.title("Change Password")

    old = st.text_input("Old password", type="password")
    new = st.text_input("New password", type="password")
    new2 = st.text_input("Confirm new password", type="password")

    if st.button("Update password", type="primary"):
        if new != new2:
            st.error("New passwords do not match.")
            return

        msg = change_password(cfg, st.session_state.auth_user, old, new)
        if msg == "Password updated.":
            st.success(msg)
        else:
            st.error(msg)


# ---------- App flow ----------
if not st.session_state.auth_user:
    login_view()
else:
    page = sidebar_controls()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Inspect & Collect":
        page_inspect_collect()
    elif page == "Status Board":
        page_status_board()
    elif page == "Change Password":
        page_change_password()