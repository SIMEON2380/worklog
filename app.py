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


# ---------- UI rules ----------
HIDE_COLS = {
    # You requested these removed from table display
    "id",
    "auth_code",
    "waiting_hours",
    "created_at",
    "hours",
    "postcode",
    "customer_name",
    "custmer_name",  # just in case the column is misspelled
    "site_address",
    "update",
}

# optional: hide common variants if your DB used spaces/case differences
HIDE_COLS |= {
    "Auth Code",
    "Waiting Hours",
    "Created At",
    "Customer Name",
    "Site Address",
    "Postcode",
    "Hours",
    "Update",
}


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
        # radio removed as requested


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


def _apply_hide_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    to_drop = [c for c in df.columns if c in HIDE_COLS]
    return df.drop(columns=to_drop, errors="ignore")


def render_coloured_table_view(df: pd.DataFrame):
    """
    Pretty view-only table with your color scheme.
    Hides columns you asked to remove from the table display.
    """
    if df.empty:
        st.write("No rows to show.")
        return

    # Keep your preferred ordering but we will hide the requested columns
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
        # plus any extra columns you may already have
    ]

    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    view = df[cols].copy()
    view = _apply_hide_cols(view)
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
    Shows a coloured view table (with hidden columns),
    and an editor where ONLY job_status is editable.
    Uses 'id' as the internal key but hides it from the UI.
    """
    if df.empty:
        st.write("No rows to show.")
        return

    if "id" not in df.columns:
        st.error("Missing required column: id (needed to save edits safely).")
        return

    st.caption("Colour view (Paid=green, Withdraw=yellow, Aborted=red)")
    render_coloured_table_view(df)

    st.divider()
    st.caption("Edit job_status below, then click Save (everything else is locked).")

    edit_df = df.copy()

    # Use id as index so it does NOT appear as a column in the editor
    edit_df = edit_df.set_index("id")

    disabled_cols = [c for c in edit_df.columns if c != "job_status"]

    # Hide columns you requested from display in the editor too
    edit_df_visible = _apply_hide_cols(edit_df)

    edited = st.data_editor(
        edit_df_visible,
        key=key,
        num_rows="fixed",
        disabled=disabled_cols,
        column_config={
            "job_status": st.column_config.SelectboxColumn("job_status", options=cfg.STATUS_OPTIONS),
        },
        use_container_width=True,
        hide_index=True,  # hides the id index
    )

    if st.button("Save status changes", key=f"{key}_save"):
        changes = 0

        # Compare against original (still indexed by id)
        orig = df.set_index("id")
        # edited currently has the same index as edit_df_visible (id index)
        new = edited.copy()

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
                    job_amount=float(full.get("amount") or 0.0),
                    job_expenses=full.get("job_expenses"),
                    expenses_amount=float(full.get("expenses_amount") or 0.0),
                    auth_code=full.get("auth_code"),
                    job_status=after,
                    waiting_time_raw=str(full.get("waiting_time") or ""),
                    comments=str(full.get("comments") or ""),
                )
                changes += 1

        st.success(f"Updated {changes} row(s).")
        st.rerun()


# ---------- Pages ----------
def page_dashboard():
    st.title("Dashboard")
    st.caption("All jobs in the database.")

    df = DB["read_all"]()
    st.subheader("All jobs (view + edit job_status)")
    editable_full_status_editor(df, key="dashboard_full_editor")


def page_inspect_collect():
    st.title("Inspect & Collect")
    st.caption("Only Inspect & Collect jobs. You can edit these and set status to Paid.")

    df_all = DB["read_all"]()
    df = df_all[df_all["category"].isin(cfg.INSPECT_COLLECT_TYPES)].copy() if not df_all.empty else df_all

    st.subheader("Inspect & Collect jobs (view + edit job_status)")
    editable_full_status_editor(df, key="inspect_full_editor")


def page_status_board():
    st.title("Status Board")
    st.caption("View jobs by status and change job statuses.")

    picked = st.selectbox("Choose a status to view", cfg.STATUS_OPTIONS, index=0)
    df_all = DB["read_all"]()
    df = df_all[df_all["job_status"] == picked].copy() if not df_all.empty else df_all

    st.subheader(f"Jobs with status: {picked} (view + edit job_status)")
    editable_full_status_editor(df, key="status_board_full_editor")


def page_settings():
    st.title("Settings")

    st.subheader("Change Password")
    old = st.text_input("Old password", type="password")
    new = st.text_input("New password", type="password")
    new2 = st.text_input("Confirm new password", type="password")

    if st.button("Update password", type="primary"):
        if new != new2:
            st.error("New passwords do not match.")
        else:
            msg = change_password(cfg, st.session_state.auth_user, old, new)
            if msg == "Password updated.":
                st.success(msg)
            else:
                st.error(msg)

    st.divider()

    st.subheader("Logout")
    if st.button("Logout"):
        st.session_state.auth_user = None
        st.rerun()


# ---------- App flow ----------
if not st.session_state.auth_user:
    login_view()
else:
    sidebar_controls()

    tabs = st.tabs(["Dashboard", "Inspect & Collect", "Status Board", "Settings"])
    with tabs[0]:
        page_dashboard()
    with tabs[1]:
        page_inspect_collect()
    with tabs[2]:
        page_status_board()
    with tabs[3]:
        page_settings()