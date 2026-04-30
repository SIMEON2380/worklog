import os
from datetime import date

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("WORKLOG_API_KEY", "")

HEADERS = {"x-api-key": API_KEY}

CATEGORY_OPTIONS = [
    "STRD Trade Plate",
    "Inspect and Collect",
    "Inspect and Collect 2",
]

JOB_STATUS_OPTIONS = [
    "Start",
    "Completed",
    "Aborted",
    "Paid",
    "Pending",
    "Withdraw",
]

JOB_OUTCOME_OPTIONS = [
    "Completed",
    "Aborted",
    "Withdraw",
    "Pending",
]


st.set_page_config(page_title="Power Page", layout="wide")
st.title("⚡ Power Page")


def safe_select_index(options, current_value, default_index=0):
    current_value = str(current_value or "").strip()
    return options.index(current_value) if current_value in options else default_index


def safe_date_value(value):
    if value is None or value == "":
        return None

    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return None

    return parsed.date()


def fetch_jobs():
    url = f"{API_URL}/jobs"
    params = {"all_records": "true", "limit": 5000}
    response = requests.get(url, headers=HEADERS, params=params, timeout=20)

    if response.status_code == 401:
        st.error("API key rejected. Check WORKLOG_API_KEY in worklog.service.")
        return pd.DataFrame()

    if response.status_code != 200:
        st.error(f"API error: {response.status_code} - {response.text}")
        return pd.DataFrame()

    payload = response.json()

    if isinstance(payload, dict):
        rows = payload.get("data", payload.get("items", []))
    else:
        rows = payload

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce")

    money_cols = [
        "amount",
        "waiting_amount",
        "expenses_amount",
        "add_pay",
        "waiting_hours",
    ]

    for col in money_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["gross_total"] = (
        df.get("amount", 0)
        + df.get("waiting_amount", 0)
        + df.get("add_pay", 0)
    )

    df["net_total"] = df["gross_total"] - df.get("expenses_amount", 0)

    return df


def api_post(path, payload):
    return requests.post(
        f"{API_URL}{path}",
        headers=HEADERS,
        json=payload,
        timeout=20,
    )


def api_put(path, payload):
    return requests.put(
        f"{API_URL}{path}",
        headers=HEADERS,
        json=payload,
        timeout=20,
    )


df = fetch_jobs()

if df.empty:
    st.warning("No jobs found.")
    st.stop()


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Daily Report",
        "Weekly Report",
        "Monthly Report",
        "Edit Job",
        "Add Job",
        "Postcode Search",
    ]
)


with tab1:
    st.subheader("Daily jobs and totals")

    daily = df.dropna(subset=["work_date"]).copy()
    daily["report_date"] = daily["work_date"].dt.date

    selected_date = st.date_input(
        "Select report date",
        value=date.today(),
        format="YYYY-MM-DD",
    )

    day_df = daily[daily["report_date"] == selected_date].copy()

    if day_df.empty:
        st.warning("No jobs found for this date.")
    else:
        total_jobs = len(day_df)
        total_amount = day_df["amount"].sum()
        total_waiting = day_df["waiting_amount"].sum()
        total_add_pay = day_df["add_pay"].sum()
        total_expenses = day_df["expenses_amount"].sum()
        total_net = day_df["net_total"].sum()

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric("Jobs", total_jobs)
        col2.metric("Amount", f"£{total_amount:,.2f}")
        col3.metric("Waiting", f"£{total_waiting:,.2f}")
        col4.metric("Add Pay", f"£{total_add_pay:,.2f}")
        col5.metric("Expenses", f"£{total_expenses:,.2f}")
        col6.metric("Net Total", f"£{total_net:,.2f}")

        daily_cols = [
            "work_date",
            "job_id",
            "category",
            "job_status",
            "amount",
            "waiting_amount",
            "add_pay",
            "expenses_amount",
            "net_total",
            "vehicle_reg",
            "vehicle_description",
            "collection_from",
            "delivery_to",
            "job_outcome",
        ]

        available_cols = [col for col in daily_cols if col in day_df.columns]

        st.dataframe(
            day_df[available_cols].sort_values("work_date", ascending=False),
            use_container_width=True,
        )


with tab2:
    st.subheader("Weekly jobs and totals")

    weekly = df.dropna(subset=["work_date"]).copy()
    weekly["week_start"] = weekly["work_date"].dt.to_period("W").apply(lambda r: r.start_time.date())
    weekly["week_end"] = weekly["work_date"].dt.to_period("W").apply(lambda r: r.end_time.date())
    weekly["week_range"] = weekly["week_start"].astype(str) + " to " + weekly["week_end"].astype(str)

    weekly_summary = (
        weekly.groupby("week_range")
        .agg(
            jobs=("id", "count"),
            amount_total=("amount", "sum"),
            waiting_total=("waiting_amount", "sum"),
            add_pay_total=("add_pay", "sum"),
            expenses_total=("expenses_amount", "sum"),
            net_total=("net_total", "sum"),
        )
        .reset_index()
        .sort_values("week_range", ascending=False)
    )

    st.dataframe(weekly_summary, use_container_width=True)


with tab3:
    st.subheader("Monthly jobs and totals")

    monthly = df.dropna(subset=["work_date"]).copy()
    monthly["month"] = monthly["work_date"].dt.to_period("M").astype(str)

    monthly_summary = (
        monthly.groupby("month")
        .agg(
            jobs=("id", "count"),
            amount_total=("amount", "sum"),
            waiting_total=("waiting_amount", "sum"),
            add_pay_total=("add_pay", "sum"),
            expenses_total=("expenses_amount", "sum"),
            net_total=("net_total", "sum"),
        )
        .reset_index()
        .sort_values("month", ascending=False)
    )

    st.dataframe(monthly_summary, use_container_width=True)


with tab4:
    st.subheader("Edit job")

    search_text = st.text_input("Search by job ID, vehicle reg, postcode, or vehicle description")

    edit_df = df.copy()

    if search_text:
        search = search_text.strip().lower()
        edit_df = edit_df[
            edit_df.astype(str)
            .apply(lambda row: row.str.lower().str.contains(search, na=False).any(), axis=1)
        ]

    display_cols = [
        "id",
        "work_date",
        "job_id",
        "category",
        "job_status",
        "amount",
        "waiting_amount",
        "vehicle_reg",
        "collection_from",
        "delivery_to",
        "job_outcome",
    ]

    st.dataframe(edit_df[display_cols], use_container_width=True)

    row_ids = edit_df["id"].dropna().astype(int).tolist()

    if row_ids:
        selected_id = st.selectbox("Select row ID to edit", row_ids)

        selected = edit_df[edit_df["id"] == selected_id].iloc[0]

        with st.form("edit_job_form"):
            work_date = st.date_input(
                "Work date",
                value=selected["work_date"].date() if pd.notna(selected["work_date"]) else date.today(),
            )

            job_id = st.text_input("Job ID", value=str(selected.get("job_id") or ""))

            category = st.selectbox(
                "Category",
                CATEGORY_OPTIONS,
                index=safe_select_index(CATEGORY_OPTIONS, selected.get("category")),
            )

            job_status = st.selectbox(
                "Job status",
                JOB_STATUS_OPTIONS,
                index=safe_select_index(JOB_STATUS_OPTIONS, selected.get("job_status")),
            )

            amount = st.number_input("Amount", value=float(selected.get("amount") or 0), step=0.01)
            waiting_time = st.text_input("Waiting time", value=str(selected.get("waiting_time") or ""))
            waiting_hours = st.number_input("Waiting hours", value=float(selected.get("waiting_hours") or 0), step=0.25)
            waiting_amount = st.number_input("Waiting amount", value=float(selected.get("waiting_amount") or 0), step=0.01)
            vehicle_description = st.text_input("Vehicle description", value=str(selected.get("vehicle_description") or ""))
            vehicle_reg = st.text_input("Vehicle reg", value=str(selected.get("vehicle_reg") or ""))
            collection_from = st.text_input("Collection from", value=str(selected.get("collection_from") or ""))
            delivery_to = st.text_input("Delivery to", value=str(selected.get("delivery_to") or ""))
            job_expenses = st.text_input("Job expenses", value=str(selected.get("job_expenses") or "No expenses"))
            expenses_amount = st.number_input("Expenses amount", value=float(selected.get("expenses_amount") or 0), step=0.01)
            auth_code = st.text_input("Auth code", value=str(selected.get("auth_code") or ""))
            comments = st.text_area("Comments", value=str(selected.get("comments") or ""))
            add_pay = st.number_input("Additional pay", value=float(selected.get("add_pay") or 0), step=0.01)

            current_paid_date = safe_date_value(selected.get("paid_date"))

            paid_date = st.date_input(
                "Paid date",
                value=current_paid_date,
                format="YYYY-MM-DD",
            )

            job_outcome = st.selectbox(
                "Job outcome",
                JOB_OUTCOME_OPTIONS,
                index=safe_select_index(JOB_OUTCOME_OPTIONS, selected.get("job_outcome")),
            )

            submitted = st.form_submit_button("Save changes")

            if submitted:
                payload = {
                    "work_date": str(work_date),
                    "job_id": job_id,
                    "category": category,
                    "job_status": job_status,
                    "amount": amount,
                    "waiting_time": waiting_time or None,
                    "waiting_hours": waiting_hours,
                    "waiting_amount": waiting_amount,
                    "vehicle_description": vehicle_description,
                    "vehicle_reg": vehicle_reg,
                    "collection_from": collection_from,
                    "delivery_to": delivery_to,
                    "job_expenses": job_expenses,
                    "expenses_amount": expenses_amount,
                    "auth_code": auth_code or None,
                    "comments": comments or None,
                    "add_pay": add_pay,
                    "paid_date": str(paid_date) if paid_date else None,
                    "job_outcome": job_outcome,
                }

                res = api_put(f"/jobs/row/{selected_id}", payload)

                if res.status_code in [200, 204]:
                    st.success("Job updated successfully.")
                    st.rerun()
                else:
                    st.error(f"Update failed: {res.status_code} - {res.text}")
    else:
        st.info("No matching jobs found.")


with tab5:
    st.subheader("Add new job")

    with st.form("add_job_form"):
        work_date = st.date_input("Work date", value=date.today())
        job_id = st.text_input("Job ID")

        category = st.selectbox(
            "Category",
            CATEGORY_OPTIONS,
        )

        job_status = st.selectbox(
            "Job status",
            JOB_STATUS_OPTIONS,
        )

        amount = st.number_input("Amount", min_value=0.0, step=0.01)
        waiting_time = st.text_input("Waiting time")
        waiting_hours = st.number_input("Waiting hours", min_value=0.0, step=0.25)
        waiting_amount = st.number_input("Waiting amount", min_value=0.0, step=0.01)
        vehicle_description = st.text_input("Vehicle description")
        vehicle_reg = st.text_input("Vehicle reg")
        collection_from = st.text_input("Collection from")
        delivery_to = st.text_input("Delivery to")
        job_expenses = st.text_input("Job expenses", value="No expenses")
        expenses_amount = st.number_input("Expenses amount", min_value=0.0, step=0.01)
        auth_code = st.text_input("Auth code")
        comments = st.text_area("Comments")
        add_pay = st.number_input("Additional pay", min_value=0.0, step=0.01)

        paid_date = st.date_input(
            "Paid date",
            value=None,
            format="YYYY-MM-DD",
        )

        job_outcome = st.selectbox(
            "Job outcome",
            JOB_OUTCOME_OPTIONS,
        )

        submitted = st.form_submit_button("Add job")

        if submitted:
            payload = {
                "work_date": str(work_date),
                "job_id": job_id,
                "category": category,
                "job_status": job_status,
                "amount": amount,
                "waiting_time": waiting_time or None,
                "waiting_hours": waiting_hours,
                "waiting_amount": waiting_amount,
                "vehicle_description": vehicle_description,
                "vehicle_reg": vehicle_reg,
                "collection_from": collection_from,
                "delivery_to": delivery_to,
                "job_expenses": job_expenses,
                "expenses_amount": expenses_amount,
                "auth_code": auth_code or None,
                "comments": comments or None,
                "add_pay": add_pay,
                "paid_date": str(paid_date) if paid_date else None,
                "job_outcome": job_outcome,
            }

            res = api_post("/jobs", payload)

            if res.status_code in [200, 201]:
                st.success("Job added successfully.")
                st.rerun()
            else:
                st.error(f"Add failed: {res.status_code} - {res.text}")


with tab6:
    st.subheader("Have I been to this postcode?")

    postcode = st.text_input("Enter postcode").strip().replace(" ", "").upper()

    if postcode:
        postcode_df = df.copy()

        postcode_df["collection_clean"] = postcode_df["collection_from"].astype(str).str.replace(" ", "", regex=False).str.upper()
        postcode_df["delivery_clean"] = postcode_df["delivery_to"].astype(str).str.replace(" ", "", regex=False).str.upper()

        matches = postcode_df[
            postcode_df["collection_clean"].str.contains(postcode, na=False)
            | postcode_df["delivery_clean"].str.contains(postcode, na=False)
        ]

        if matches.empty:
            st.warning("No, you have not been to this postcode before.")
        else:
            st.success(f"Yes, you have been to this postcode {len(matches)} time(s).")
            st.dataframe(
                matches[
                    [
                        "work_date",
                        "job_id",
                        "vehicle_reg",
                        "vehicle_description",
                        "collection_from",
                        "delivery_to",
                        "amount",
                        "waiting_amount",
                        "net_total",
                        "job_status",
                        "job_outcome",
                    ]
                ],
                use_container_width=True,
            )