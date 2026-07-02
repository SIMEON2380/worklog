import re
from datetime import date

import pandas as pd
import streamlit as st

from worklog.auth import ensure_default_user
from worklog.config import Config
from worklog.db import make_db
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


def time_since_visit(last_date) -> str:
    if pd.isna(last_date):
        return "N/A"

    today = pd.Timestamp(date.today())
    last_date = pd.to_datetime(last_date, errors="coerce")

    if pd.isna(last_date):
        return "N/A"

    last_date = last_date.normalize()

    # If the stored job date is ahead of today's system date,
    # avoid showing a confusing "Future date" message.
    if last_date > today:
        return "Today"

    total_days = int((today - last_date).days)

    if total_days == 0:
        return "Today"

    months = total_days // 30
    remaining_days = total_days % 30
    weeks = remaining_days // 7
    days = remaining_days % 7

    parts = []

    if months:
        parts.append(f"{months} month{'s' if months != 1 else ''}")

    if weeks:
        parts.append(f"{weeks} week{'s' if weeks != 1 else ''}")

    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")

    return ", ".join(parts) + " ago"


def find_postcode_matches(df: pd.DataFrame, postcode: str) -> pd.DataFrame:
    if df.empty or not postcode:
        return pd.DataFrame()

    target = normalise_postcode(postcode)

    if not target:
        return pd.DataFrame()

    sub = df.copy()

    candidate_cols = [
        c for c in ["postcode", "collection_from", "delivery_to"]
        if c in sub.columns
    ]

    if not candidate_cols:
        return pd.DataFrame()

    mask = pd.Series(False, index=sub.index)

    for col in candidate_cols:
        norm_col = f"{col}_norm"
        sub[norm_col] = sub[col].fillna("").astype(str).apply(normalise_postcode)

        # Exact match only.
        # This prevents loose postcode matches from producing wrong results.
        mask = mask | (sub[norm_col] == target)

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
    st.success("New location. No previous jobs found for this postcode.")
    st.stop()


times_visited = len(matches)

last_visited = "N/A"
time_since_last_visit = "N/A"

if "work_date" in matches.columns and matches["work_date"].notna().any():
    last_date = matches["work_date"].dropna().iloc[0]
    last_visited = last_date.strftime("%Y-%m-%d")
    time_since_last_visit = time_since_visit(last_date)


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


m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Times Visited", times_visited)
m2.metric("Last Visited", last_visited)
m3.metric("Time Since Last Visit", time_since_last_visit)
m4.metric("Last Vehicle", last_vehicle)
m5.metric("Last Job Type", last_job_type)

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

st.dataframe(
    matches[available_cols],
    use_container_width=True,
    hide_index=True,
)