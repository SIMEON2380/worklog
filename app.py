import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user, verify_login
from worklog.ui import display_jobs_table
from worklog.api import fetch_jobs

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=cfg.APP_TITLE, layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None


def render_login():
    st.title(cfg.APP_TITLE)
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = verify_login(cfg, username, password)
        if user:
            st.session_state.auth_user = user
            st.success("Logged in")
            st.rerun()
        else:
            st.error("Invalid credentials")


if not st.session_state.auth_user:
    render_login()
    st.stop()

st.title("Dashboard")
st.caption("All jobs in the database")

try:
    jobs = fetch_jobs()
    if isinstance(jobs, str):
        st.error(f"Failed to load jobs from API: {jobs}")
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(jobs)
except Exception as e:
    st.error(f"Failed to load jobs from API: {e}")
    df = pd.DataFrame()

display_jobs_table(cfg, df, caption=None)