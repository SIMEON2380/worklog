import os

import pandas as pd
import requests
import streamlit as st

from worklog.auth import ensure_default_user, verify_login
from worklog.config import Config
from worklog.db import make_db
from worklog.ui import display_jobs_table

cfg = Config()
DB = make_db(cfg)

API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("WORKLOG_API_KEY") or os.getenv("API_KEY")

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


def fetch_jobs():
    headers = {"x-api-key": API_KEY} if API_KEY else {}

    response = requests.get(
        f"{API_URL}/jobs",
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()

    payload = response.json()

    if isinstance(payload, dict):
        if "data" in payload:
            return payload["data"]
        if "jobs" in payload:
            return payload["jobs"]

    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unexpected API response format: {payload}")


if not st.session_state.auth_user:
    render_login()
    st.stop()

st.title("Dashboard")
st.caption("All jobs in the database")

try:
    jobs = fetch_jobs()
    df = pd.DataFrame(jobs)
except Exception as e:
    st.error(f"Failed to load jobs from API: {e}")
    df = pd.DataFrame()

display_jobs_table(cfg, df, caption=None)