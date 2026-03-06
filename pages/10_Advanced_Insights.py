import streamlit as st
import pandas as pd

from worklog.config import Config
from worklog.db import make_db
from worklog.auth import ensure_default_user
from worklog.ui import require_login

cfg = Config()
DB = make_db(cfg)

st.set_page_config(page_title=f"{cfg.APP_TITLE} - Advanced Insights", layout="wide")

DB["ensure_schema"]()
ensure_default_user(cfg)
require_login()

st.subheader("Advanced Insights")

df = DB["read_all"]().copy()

if df.empty:
    st.info("No jobs found.")
    st.stop()

if "vehicle_description" not in df.columns:
    st.error("Missing 'vehicle_description' column in dataset.")
    st.stop()

if "work_date" in df.columns:
    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce")

# -------------------------
# Clean vehicle description
# -------------------------
df["vehicle_description"] = (
    df["vehicle_description"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["vehicle_description"] != ""]

if df.empty:
    st.info("No vehicle descriptions found.")
    st.stop()

# -------------------------
# Improved brand extractor
# -------------------------
KNOWN_MAKES = {
    "TESLA": ["TESLA"],
    "BMW": ["BMW"],
    "AUDI": ["AUDI"],
    "MERCEDES": ["MERCEDES", "MERC"],
    "VOLKSWAGEN": ["VOLKSWAGEN", "VW"],
    "FORD": ["FORD"],
    "TOYOTA": ["TOYOTA"],
    "NISSAN": ["NISSAN"],
    "KIA": ["KIA"],
    "HYUNDAI": ["HYUNDAI"],
    "PEUGEOT": ["PEUGEOT"],
    "RENAULT": ["RENAULT"],
    "VAUXHALL": ["VAUXHALL"],
    "SKODA": ["SKODA"],
    "SEAT": ["SEAT"],
    "LAND ROVER": ["LAND ROVER"],
    "RANGE ROVER": ["RANGE ROVER"],
    "VOLVO": ["VOLVO"],
    "MINI": ["MINI"],
    "HONDA": ["HONDA"],
    "MAZDA": ["MAZDA"],
    "LEXUS": ["LEXUS"],
    "PORSCHE": ["PORSCHE"],
    "JAGUAR": ["JAGUAR"],
    "FIAT": ["FIAT"],
    "CITROEN": ["CITROEN"],
    "DACIA": ["DACIA"],
    "MG": ["MG"],
    "BYD": ["BYD"],
    "CUPRA": ["CUPRA"],
}

def extract_make(desc: str) -> str:
    text = str(desc).strip().upper()
    if not text:
        return "UNKNOWN"

    for brand, patterns in KNOWN_MAKES.items():
        for pattern in patterns:
            if pattern in text:
                return brand

    words = text.split()
    if not words:
        return "UNKNOWN"

    return words[0]

df["vehicle_make"] = df["vehicle_description"].apply(extract_make)

# -------------------------
# Filters
# -------------------------
c1, c2 = st.columns(2)

all_makes = sorted(df["vehicle_make"].dropna().unique().tolist())
selected_make = c1.selectbox("Filter by make", ["All"] + all_makes, index=0)

search_text = c2.text_input("Search vehicle description or make")

filtered = df.copy()

if selected_make != "All":
    filtered = filtered[filtered["vehicle_make"] == selected_make]

if search_text.strip():
    q = search_text.strip().upper()
    filtered = filtered[
        filtered["vehicle_make"].str.upper().str.contains(q, na=False)
        | filtered["vehicle_description"].str.upper().str.contains(q, na=False)
    ]

if filtered.empty:
    st.info("No matching vehicles found.")
    st.stop()

# -------------------------
# Summary metrics
# -------------------------
total_vehicle_jobs = len(filtered)
unique_vehicle_desc = filtered["vehicle_description"].nunique()
unique_makes = filtered["vehicle_make"].nunique()
top_make = (
    filtered["vehicle_make"].value_counts().idxmax()
    if not filtered["vehicle_make"].empty
    else "N/A"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Vehicle Jobs", f"{total_vehicle_jobs}")
m2.metric("Unique Descriptions", f"{unique_vehicle_desc}")
m3.metric("Unique Makes", f"{unique_makes}")
m4.metric("Top Make", top_make)

st.divider()

# -------------------------
# Top makes
# -------------------------
st.markdown("### Most driven makes")

make_counts = (
    filtered["vehicle_make"]
    .value_counts()
    .rename_axis("Make")
    .reset_index(name="Jobs")
)

st.dataframe(make_counts, use_container_width=True, hide_index=True)

# -------------------------
# Top exact vehicle descriptions
# -------------------------
st.markdown("### Most driven vehicle descriptions")

desc_counts = (
    filtered["vehicle_description"]
    .value_counts()
    .rename_axis("Vehicle Description")
    .reset_index(name="Jobs")
)

st.dataframe(desc_counts.head(30), use_container_width=True, hide_index=True)

# -------------------------
# Specific search result block
# -------------------------
if search_text.strip():
    st.markdown("### Search result summary")
    last_driven = None
    if "work_date" in filtered.columns and filtered["work_date"].notna().any():
        last_driven = filtered["work_date"].max()

    s1, s2 = st.columns(2)
    s1.metric("Matching Jobs", f"{len(filtered)}")
    s2.metric(
        "Last Driven",
        last_driven.strftime("%Y-%m-%d") if pd.notna(last_driven) else "N/A"
    )

# -------------------------
# Optional raw records
# -------------------------
with st.expander("Show matching jobs"):
    cols_to_drop = ["vehicle_make"]
    raw = filtered.drop(columns=cols_to_drop, errors="ignore")
    st.dataframe(raw, use_container_width=True)