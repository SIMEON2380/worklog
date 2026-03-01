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


# ---------- Metrics ----------
def show_top_metrics_from_old_schema(df: pd.DataFrame):
    if df.empty:
        st.info("No jobs found.")
        return

    total_jobs = int(len(df))
    total_job = pd.to_numeric(df["amount"], errors="coerce").fillna(0).sum()
    total_exp = pd.to_numeric(df["expenses_amount"], errors="coerce").fillna(0).sum()
    total_wait = pd.to_numeric(df["waiting_amount"], errors="coerce").fillna(0).sum()

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


# ---------- Styling (your colour scheme) ----------
def _row_status_style(row: pd.Series):
    status = str(row.get("job_status", "")).strip().lower()
    base = "color: #111; font-weight: 600;"
    if status == "paid":
        return [base + "background-color: #d1fae5;"] * len(row)  # green
    if status == "withdraw":
        return [base + "background-color: #fde68a;"] * len(row)  # yellow
    if status == "aborted":
        return [base + "background-color: #fecaca;"] * len(row)  # red
    return [base] * len(row)


def _format_money_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["amount", "expenses_amount", "waiting_amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def render_coloured_table_view(df: pd.DataFrame):
    """
    Pretty view-only table with your color scheme.
    Uses the original column names and shows everything, including comments/locations.
    """
    if df.empty:
        st.write("No rows to show.")
        return

    # Put most important columns first, but keep everything available
    preferred_order = [
        "id",
        "work_date",
        "job_id",
        "category",
        "job_status",
        "vehicle_description",
        "vehicle_reg",
        "collection_from",
        "delivery_to",
        "amount",
        "job_expenses",
        "expenses_amount",
        "auth_code",
        "waiting_time",
        "waiting_hours",
        "waiting_amount",
        "comments",
        "created_at",
    ]

    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    view = df[cols].copy()
    view = _format_money_cols(view)

    sty = view.style.apply(_row_status_style, axis=1)

    fmt = {}
    if "amount" in view.columns:
        fmt["amount"] = lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
    if "expenses_amount" in view.columns:
        fmt["expenses_amount"] = lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
    if "waiting_amount" in view.columns:
        fmt["waiting_amount"] = lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
    if fmt:
        sty = sty.format(fmt)

    st.dataframe(sty, use_container_width=True, hide_index=True)


def editable_full_status_editor(df: pd.DataFrame, key: str):
    """
    Full table visible, only job_status editable.
    Also shows a coloured view table above for the nice look.
    """
    if df.empty:
        st.write("No rows to show.")
        return

    st.caption("Colour view (Paid=green, Withdraw=yellow, Aborted=red)")
    render_coloured_table_view(df)

    st.divider()
    st.caption("Edit job_status below, then click Save (everything else is locked).")

    # Editor dataframe: keep original names, keep all columns
    edit_df = df.copy()

    disabled_cols = [c for c in edit_df.columns if c != "job_status"]

    edited = st.data_editor(
        edit_df,
        key=key,
        num_rows="fixed",
        disabled=disabled_cols,
        column_config={
            "job_status": st.column_config.SelectboxColumn("job_status", options=cfg.STATUS_OPTIONS),
        },
        use_container_width=True,
    )

    if st.button("Save status changes", key=f"{key}_save"):
        changes = 0

        orig = df.set_index("id")
        new = edited.set_index("id")

        for row_id in new.index:
            before = str(orig.loc[row_id, "job_status"])
            after = str(new.loc[row_id, "job_status"])
            if before != after:
                full = df[df["id"] == row_id].iloc[0].to_dict()

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

    c1, c2, c3, c4 = st.columns(4)
    date_from = c1.date_input("From", value=None)
    date_to = c2.date_input("To", value=None)
    status = c3.selectbox("Status", ["All"] + cfg.STATUS_OPTIONS, index=0)
    job_type = c4.selectbox("Job type", ["All"] + cfg.JOB_TYPE_OPTIONS, index=0)
    search = st.text_input("Search (job / reg / auth / locations / comments)", placeholder="type anything...").strip()

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

    st.subheader("All jobs (view + edit job_status)")
    editable_full_status_editor(df, key="dashboard_full_editor")


def page_inspect_collect():
    st.title("Inspect & Collect")
    st.caption("Only Inspect & Collect jobs. You can edit these and set status to Paid.")

    df_all = DB["read_all"]()
    df = df_all[df_all["category"].isin(cfg.INSPECT_COLLECT_TYPES)].copy() if not df_all.empty else df_all

    show_top_metrics_from_old_schema(df)
    st.subheader("Inspect & Collect jobs (view + edit job_status)")
    editable_full_status_editor(df, key="inspect_full_editor")


def page_status_board():
    st.title("Status Board")
    st.caption("View jobs by status and change job statuses.")

    picked = st.selectbox("Choose a status to view", cfg.STATUS_OPTIONS, index=0)
    df_all = DB["read_all"]()
    df = df_all[df_all["job_status"] == picked].copy() if not df_all.empty else df_all

    show_top_metrics_from_old_schema(df)

    st.subheader(f"Jobs with status: {picked} (view + edit job_status)")
    editable_full_status_editor(df, key="status_board_full_editor")


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