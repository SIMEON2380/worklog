import re
import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Postcode Lookup", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Postcode Lookup")


def normalise_postcode(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    return text


def find_postcode_matches(df: pd.DataFrame, postcode: str) -> pd.DataFrame:
    if df.empty or not postcode:
        return pd.DataFrame()

    target = normalise_postcode(postcode)
    sub = df.copy()

    candidate_cols = [c for c in ["postcode", "collection_from", "delivery_to"] if c in sub.columns]
    if not candidate_cols:
        return pd.DataFrame()

    for col in candidate_cols:
        sub[f"{col}_norm"] = sub[col].fillna("").astype(str).apply(normalise_postcode)

    mask = False
    for col in candidate_cols:
        norm_col = f"{col}_norm"
        mask = mask | sub[norm_col].str.contains(target, na=False)

    matches = sub[mask].copy()

    if matches.empty:
        return matches

    if "work_date" in matches.columns:
        matches["work_date"] = pd.to_datetime(matches["work_date"], errors="coerce")
        matches = matches.sort_values("work_date", ascending=False)

    return matches


df = DB["read_all"]().copy()

if df.empty:
    st.info("No jobs found.")
    st.stop()

postcode_query = st.text_input("Enter postcode")

if not postcode_query.strip():
    st.info("Enter a postcode to search your history.")
    st.stop()

matches = find_postcode_matches(df, postcode_query)

if matches.empty:
    st.info("No previous jobs found for this postcode.")
    st.stop()

times_visited = len(matches)

last_visited = "N/A"
if "work_date" in matches.columns and matches["work_date"].notna().any():
    last_visited = matches["work_date"].dropna().iloc[0].strftime("%Y-%m-%d")

last_vehicle = "N/A"
if "vehicle_description" in matches.columns:
    vals = matches["vehicle_description"].fillna("").astype(str).str.strip()
    vals = vals[vals != ""]
    if not vals.empty:
        last_vehicle = vals.iloc[0]

last_job_type = "N/A"
if "category" in matches.columns:
    vals = matches["category"].fillna("").astype(str).str.strip()
    vals = vals[vals != ""]
    if not vals.empty:
        last_job_type = vals.iloc[0]

last_comment = "N/A"
comment_col = None
for col in ["comments", "description", "comment", "notes"]:
    if col in matches.columns:
        comment_col = col
        break

if comment_col:
    vals = matches[comment_col].fillna("").astype(str).str.strip()
    vals = vals[vals != ""]
    if not vals.empty:
        last_comment = vals.iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Times Visited", times_visited)
m2.metric("Last Visited", last_visited)
m3.metric("Last Vehicle", last_vehicle)
m4.metric("Last Job Type", last_job_type)

if last_comment != "N/A":
    st.caption(f"Last note: {last_comment}")

st.divider()

st.markdown("### Matching jobs")

show_cols = [
    "work_date",
    "job_id",
    "category",
    "vehicle_description",
    "vehicle_reg",
    "collection_from",
    "delivery_to",
    "postcode",
    "job_status",
    "comments",
]

available_cols = [c for c in show_cols if c in matches.columns]
st.dataframe(matches[available_cols], use_container_width=True, hide_index=True)