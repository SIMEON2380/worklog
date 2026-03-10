import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Driver Intelligence", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Driver Intelligence")

df = DB["read_all"]().copy()

if df.empty:
    st.info("No jobs found.")
    st.stop()


# -------------------------
# Helpers
# -------------------------
def safe_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def get_status_col(frame: pd.DataFrame) -> str | None:
    if "job_status" in frame.columns:
        return "job_status"
    if "status" in frame.columns:
        return "status"
    return None


def exclude_withdraw(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    status_col = get_status_col(out)
    if status_col:
        out[status_col] = safe_text(out[status_col]).str.lower()
        out = out[out[status_col] != "withdraw"].copy()
    return out


def build_driver_pay(frame: pd.DataFrame, waiting_rate: float = 7.5) -> pd.DataFrame:
    out = frame.copy()

    out["amount"] = safe_num(out["amount"]) if "amount" in out.columns else 0.0
    out["waiting_hours"] = safe_num(out["waiting_hours"]) if "waiting_hours" in out.columns else 0.0
    out["waiting_amount"] = safe_num(out["waiting_amount"]) if "waiting_amount" in out.columns else 0.0
    out["add_pay"] = safe_num(out["add_pay"]) if "add_pay" in out.columns else 0.0

    computed_wait = out["waiting_hours"] * float(waiting_rate)
    out["waiting_amount"] = out["waiting_amount"].where(out["waiting_amount"] != 0, computed_wait)

    out["driver_pay"] = out["amount"] + out["waiting_amount"] + out["add_pay"]
    return out


def top_counts(frame: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    if col not in frame.columns:
        return pd.DataFrame(columns=[col, "count"])
    s = safe_text(frame[col])
    s = s[s != ""]
    if s.empty:
        return pd.DataFrame(columns=[col, "count"])
    out = s.value_counts().head(top_n).reset_index()
    out.columns = [col, "count"]
    return out


def top_avg_pay(frame: pd.DataFrame, group_col: str, top_n: int = 10) -> pd.DataFrame:
    if group_col not in frame.columns:
        return pd.DataFrame(columns=[group_col, "jobs", "avg_pay", "total_pay"])

    temp = frame.copy()
    temp[group_col] = safe_text(temp[group_col])
    temp = temp[temp[group_col] != ""].copy()

    if temp.empty:
        return pd.DataFrame(columns=[group_col, "jobs", "avg_pay", "total_pay"])

    out = (
        temp.groupby(group_col, dropna=False)
        .agg(
            jobs=("driver_pay", "size"),
            avg_pay=("driver_pay", "mean"),
            total_pay=("driver_pay", "sum"),
        )
        .reset_index()
        .sort_values(["avg_pay", "jobs"], ascending=[False, False])
        .head(top_n)
    )

    out["avg_pay"] = out["avg_pay"].round(2)
    out["total_pay"] = out["total_pay"].round(2)
    return out


# -------------------------
# Prep data
# -------------------------
df = exclude_withdraw(df)
df = build_driver_pay(df, waiting_rate=7.5)

if "work_date" in df.columns:
    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce")
    df["weekday"] = df["work_date"].dt.day_name()
else:
    df["weekday"] = ""

for col in [
    "vehicle_description",
    "vehicle_reg",
    "collection_from",
    "delivery_to",
    "category",
    "comments",
    "notes",
]:
    if col not in df.columns:
        df[col] = ""

df["vehicle_description"] = safe_text(df["vehicle_description"])
df["vehicle_reg"] = safe_text(df["vehicle_reg"])
df["collection_from"] = safe_text(df["collection_from"])
df["delivery_to"] = safe_text(df["delivery_to"])
df["category"] = safe_text(df["category"])

note_col = "comments" if "comments" in df.columns else "notes"
df[note_col] = safe_text(df[note_col])

# combined searchable location text
df["_location_text"] = (
    safe_text(df["collection_from"]) + " " + safe_text(df["delivery_to"])
).str.strip()


# -------------------------
# KPIs
# -------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Jobs", int(len(df)))
c2.metric("Total Driver Pay", f"£{df['driver_pay'].sum():,.2f}")
c3.metric("Unique Vehicles", int(df["vehicle_description"][df["vehicle_description"] != ""].nunique()))
c4.metric("Unique Locations", int(df["_location_text"][df["_location_text"] != ""].nunique()))

st.divider()

# -------------------------
# Main analytics
# -------------------------
left, right = st.columns(2)

with left:
    st.markdown("### Top Vehicles Driven")
    vehicles = top_counts(df, "vehicle_description", top_n=10)
    if vehicles.empty:
        st.info("No vehicle data found.")
    else:
        st.dataframe(vehicles, use_container_width=True, hide_index=True)

    st.markdown("### Most Common Collection Locations")
    collections = top_counts(df, "collection_from", top_n=10)
    if collections.empty:
        st.info("No collection location data found.")
    else:
        st.dataframe(collections, use_container_width=True, hide_index=True)

    st.markdown("### Best Paying Job Types")
    job_types = top_avg_pay(df, "category", top_n=10)
    if job_types.empty:
        st.info("No job type data found.")
    else:
        st.dataframe(job_types, use_container_width=True, hide_index=True)

with right:
    st.markdown("### Top Vehicle Regs")
    regs = top_counts(df, "vehicle_reg", top_n=10)
    if regs.empty:
        st.info("No vehicle reg data found.")
    else:
        st.dataframe(regs, use_container_width=True, hide_index=True)

    st.markdown("### Most Common Delivery Locations")
    deliveries = top_counts(df, "delivery_to", top_n=10)
    if deliveries.empty:
        st.info("No delivery location data found.")
    else:
        st.dataframe(deliveries, use_container_width=True, hide_index=True)

    st.markdown("### Best Earning Weekdays")
    weekdays_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    wd = df[df["weekday"].isin(weekdays_order)].copy()

    if wd.empty:
        st.info("No weekday data found.")
    else:
        weekday_stats = (
            wd.groupby("weekday")
            .agg(
                jobs=("driver_pay", "size"),
                avg_pay=("driver_pay", "mean"),
                total_pay=("driver_pay", "sum"),
            )
            .reset_index()
        )
        weekday_stats["weekday"] = pd.Categorical(
            weekday_stats["weekday"], categories=weekdays_order, ordered=True
        )
        weekday_stats = weekday_stats.sort_values("weekday")
        weekday_stats["avg_pay"] = weekday_stats["avg_pay"].round(2)
        weekday_stats["total_pay"] = weekday_stats["total_pay"].round(2)

        st.dataframe(weekday_stats, use_container_width=True, hide_index=True)

st.divider()

# -------------------------
# Place / postcode memory lookup
# -------------------------
st.markdown("### Have I been here before?")

search_term = st.text_input(
    "Enter postcode, town, place, or part of an address",
    placeholder="e.g. LS12 4AB or Birmingham"
).strip()

if search_term:
    s = search_term.lower()

    hits = df[
        df["collection_from"].str.lower().str.contains(s, na=False)
        | df["delivery_to"].str.lower().str.contains(s, na=False)
        | df["_location_text"].str.lower().str.contains(s, na=False)
    ].copy()

    if hits.empty:
        st.warning("No previous jobs found for that place.")
    else:
        hits = hits.sort_values("work_date", ascending=False)

        a1, a2, a3 = st.columns(3)
        a1.metric("Times Visited", int(len(hits)))
        a2.metric("Total Earned", f"£{hits['driver_pay'].sum():,.2f}")
        a3.metric("Average Earned", f"£{hits['driver_pay'].mean():,.2f}")

        latest = hits.iloc[0]

        st.markdown("#### Latest Visit")
        latest_date = ""
        if pd.notna(latest.get("work_date")):
            latest_date = pd.to_datetime(latest["work_date"]).strftime("%Y-%m-%d")

        st.write(f"**Date:** {latest_date or '-'}")
        st.write(f"**Vehicle:** {latest.get('vehicle_description', '') or '-'}")
        st.write(f"**Reg:** {latest.get('vehicle_reg', '') or '-'}")
        st.write(f"**Collection:** {latest.get('collection_from', '') or '-'}")
        st.write(f"**Delivery:** {latest.get('delivery_to', '') or '-'}")
        st.write(f"**Driver Pay:** £{float(latest.get('driver_pay', 0.0)):,.2f}")

        latest_note = latest.get(note_col, "")
        if latest_note:
            st.write(f"**Notes:** {latest_note}")

        show_cols = [
            "work_date",
            "vehicle_description",
            "vehicle_reg",
            "collection_from",
            "delivery_to",
            "category",
            "driver_pay",
            note_col,
        ]
        show_cols = [c for c in show_cols if c in hits.columns]

        display_hits = hits[show_cols].copy()
        if "work_date" in display_hits.columns:
            display_hits["work_date"] = pd.to_datetime(display_hits["work_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "driver_pay" in display_hits.columns:
            display_hits["driver_pay"] = pd.to_numeric(display_hits["driver_pay"], errors="coerce").fillna(0).round(2)

        st.markdown("#### Previous Jobs For This Place")
        st.dataframe(display_hits, use_container_width=True, hide_index=True)