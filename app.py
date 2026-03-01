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


# ---------- Small UI helpers (kept simple + stable) ----------
def show_top_metrics_from_old_schema(df: pd.DataFrame):
    if df.empty:
        st.info("No jobs found.")
        return

    # Your schema names
    total_jobs = int(len(df))
    total_job = pd.to_numeric(df["amount"], errors="coerce").fillna(0).sum()
    total_exp = pd.to_numeric(df["expenses_amount"], errors="coerce").fillna(0).sum()
    total_wait = pd.to_numeric(df["waiting_amount"], errors="coerce").fillna(0).sum()

    # Inspect & Collect pay (matches your old logic)
    df_money = df[df["job_status"].astype(str).str.lower().str.strip() != "withdraw"].copy()
    ic_jobs = df_money[df_money["category"].isin(cfg.INSPECT_COLLECT_TYPES)]
    ic_pay_total = float(len(ic_jobs) * cfg.INSPECT_COLLECT_RATE)

    total_owed = float(total_job + total_exp + total_wait + ic_pay_total)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Jobs", f"{total_jobs:,}")
    c2.metric("Job amount", f"£{total_job:,.2f}")
    c3.metric("Expenses", f"£{total_exp:,.2f}")
    c4.metric("Waiting owed", f"£{total_wait:,.2f}")
    c5.metric("Total owed", f"£{total_owed:,.2f}")


def status_bar_from_old_schema(df: pd.DataFrame):
    if df.empty:
        return
    counts = df["job_status"].fillna("Unknown").value_counts()
    st.subheader("Status overview")
    st.bar_chart(counts)


def editable_status_table_old_schema(df: pd.DataFrame, key: str):
    """
    Minimal editor that only lets you change job_status (safe + won't break DB).
    Uses update_row_by_id so waiting calc etc stays consistent.
    """
    if df.empty:
        st.write("No rows to show.")
        return

    # Only show key columns to reduce risk
    view = df[["id", "work_date", "job_id", "category", "vehicle_reg", "auth_code", "job_status"]].copy()
    view = view.rename(columns={"job_status": "status"})

    edited = st.data_editor(
        view,
        key=key,
        num_rows="fixed",
        disabled=["id", "work_date", "job_id", "category", "vehicle_reg", "auth_code"],
        column_config={
            "status": st.column_config.SelectboxColumn("status", options=cfg.STATUS_OPTIONS),
        },
        use_container_width=True,
    )

    if st.button("Save status changes", key=f"{key}_save"):
        changes = 0

        orig = view.set_index("id")
        new = edited.set_index("id")

        for row_id in new.index:
            before = str(orig.loc[row_id, "status"])
            after = str(new.loc[row_id, "status"])
            if before != after:
                # Load full row so we can call update_row_by_id safely
                full = df[df["id"] == row_id].iloc[0].to_dict()

                # Keep everything identical except status
                DB["update_row_by_id"](
                    row_id=int(row_id),
                    work_date_val=full["work_date"],
                    job_number=full["job_id"],
                    job_type=full["category"],
                    vehicle_description=full["vehicle_description"],
                    vehicle_reg=full["vehicle_reg"],
                    collection_from=full["collection_from"],
                    delivery_to=full["delivery_to"],
                    job_amount=float(full["amount"] or 0.0),
                    job_expenses=full["job_expenses"],
                    expenses_amount=float(full["expenses_amount"] or 0.0),
                    auth_code=full["auth_code"],
                    job_status=after,
                    waiting_time_raw=str(full["waiting_time"] or ""),
                    comments=str(full["comments"] or ""),
                )
                changes += 1

        st.success(f"Updated {changes} row(s).")
        st.rerun()


# ---------- Pages ----------
def page_dashboard():
    st.title("Dashboard")
    st.caption("All jobs in the database.")

    df_all = DB["read_all"]()

    # Filters (basic)
    c1, c2, c3, c4 = st.columns(4)
    date_from = c1.date_input("From", value=None)
    date_to = c2.date_input("To", value=None)
    status = c3.selectbox("Status", ["All"] + cfg.STATUS_OPTIONS, index=0)
    job_type = c4.selectbox("Job type", ["All"] + cfg.JOB_TYPE_OPTIONS, index=0)
    search = st.text_input("Search (job / reg / auth / locations)", placeholder="type anything...").strip()

    df = df_all.copy()
    if not df.empty:
        df = df[df["work_date"].notna()].copy()

        if date_from:
            df = df[df["work_date"] >= date_from]
        if date_to:
            df = df[df["work_date"] <= date_to]
        if status != "All":
            df = df[df["job_status"] == status]
        if job_type != "All":
            df = df[df["category"] == job_type]

        if search:
            mask = (
                df["job_id"].astype(str).str.contains(search, case=False, na=False)
                | df["vehicle_reg"].astype(str).str.contains(search, case=False, na=False)
                | df["auth_code"].astype(str).str.contains(search, case=False, na=False)
                | df["collection_from"].astype(str).str.contains(search, case=False, na=False)
                | df["delivery_to"].astype(str).str.contains(search, case=False, na=False)
                | df["comments"].astype(str).str.contains(search, case=False, na=False)
            )
            df = df[mask]

    show_top_metrics_from_old_schema(df)
    status_bar_from_old_schema(df)

    st.subheader("All jobs (edit status)")
    editable_status_table_old_schema(df, key="dashboard_status_editor")


def page_inspect_collect():
    st.title("Inspect & Collect")
    st.caption("Only Inspect & Collect jobs. You can edit these and set status to Paid.")

    df_all = DB["read_all"]()
    df = df_all[df_all["category"].isin(cfg.INSPECT_COLLECT_TYPES)].copy() if not df_all.empty else df_all

    show_top_metrics_from_old_schema(df)
    st.subheader("Inspect & Collect jobs (edit status)")
    editable_status_table_old_schema(df, key="inspect_status_editor")


def page_status_board():
    st.title("Status Board")
    st.caption("View jobs by status and change job statuses.")

    picked = st.selectbox("Choose a status to view", cfg.STATUS_OPTIONS, index=0)
    df_all = DB["read_all"]()
    df = df_all[df_all["job_status"] == picked].copy() if not df_all.empty else df_all

    show_top_metrics_from_old_schema(df)

    st.subheader(f"Jobs with status: {picked} (edit status)")
    editable_status_table_old_schema(df, key="status_board_editor")


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