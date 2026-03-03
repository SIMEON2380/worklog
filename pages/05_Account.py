
import streamlit as st

from worklog.config import Config
from worklog.auth import change_password
from worklog.ui import require_login

cfg = Config()
st.set_page_config(page_title=f"{cfg.APP_TITLE} - Account", layout="wide")

require_login()

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
            ok, msg = change_password(cfg, st.session_state.auth_user, current, new)
            if ok:
                st.success("Password changed.")
            else:
                st.error(msg)

if st.button("Logout"):
    st.session_state.auth_user = None
    st.rerun()
