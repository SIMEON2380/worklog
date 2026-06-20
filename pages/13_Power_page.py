import json
import os
from datetime import date

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000").rstrip("/")
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

EXPENSE_OPTIONS = [
    "No expenses",
    "Fuel",
    "Train",
    "Taxi",
    "Bus",
    "Parking",
    "Toll",
    "Hotel",
    "Food",
    "Other",
]

WAITING_RATE = float(os.getenv("WORKLOG_WAITING_RATE", "10"))

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


def clean_text(value):
    value = str(value or "").strip()
    return value if value else None


def clean_lower(value):
    return str(value or "").strip().lower()


def clean_postcode(value):
    return str(value or "").replace(" ", "").strip().upper()


def ensure_column(df, column, default_value=""):
    if column not in df.columns:
        df[column] = default_value
    return df


def parse_wait_range_to_hours(s: str) -> float:
    if not s:
        return 0.0

    s = str(s).strip().replace(" ", "")

    if "-" not in s:
        return 0.0

    start, end = s.split("-", 1)

    def to_minutes(t: str) -> int:
        if ":" in t:
            hh, mm = t.split(":", 1)
            return int(hh) * 60 + int(mm)
        return int(t) * 60

    try:
        a = to_minutes(start)
        b = to_minutes(end)

        if b <= a:
            return 0.0

        return round((b - a) / 60.0, 2)
    except Exception:
        return 0.0


def parse_expenses_for_edit(expense_text, fallback_amount=0.0):
    if not expense_text or str(expense_text).strip().lower() == "no expenses":
        return [{"type": "No expenses", "amount": 0.0}]

    rows = []

    for part in str(expense_text).split(";"):
        part = part.strip()

        if not part:
            continue

        if ":" in part:
            expense_type, amount_text = part.split(":", 1)
            expense_type = expense_type.strip()
            amount_text = amount_text.replace("£", "").replace(",", "").strip()

            try:
                amount = float(amount_text)
            except Exception:
                amount = 0.0

            rows.append({"type": expense_type, "amount": amount})

    if not rows and float(fallback_amount or 0.0) > 0:
        rows.append({"type": "Other", "amount": float(fallback_amount)})

    return rows or [{"type": "No expenses", "amount": 0.0}]


def build_expenses_from_rows(rows):
    valid_rows = [
        row
        for row in rows
        if row.get("type") != "No expenses" and float(row.get("amount") or 0.0) > 0
    ]

    total = round(sum(float(row["amount"]) for row in valid_rows), 2)

    expense_text = (
        "; ".join(
            f"{row['type']}: £{float(row['amount']):.2f}"
            for row in valid_rows
        )
        if valid_rows
        else "No expenses"
    )

    return expense_text, total, valid_rows


def parse_json_if_needed(value):
    if isinstance(value, str):
        value = value.strip()

        if value.startswith("{") and value.endswith("}"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

    return value


def extract_rows_from_payload(payload):
    payload = parse_json_if_needed(payload)

    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    for key in ["data", "items", "results", "rows"]:
        value = payload.get(key)
        value = parse_json_if_needed(value)

        if isinstance(value, list):
            return value

        if isinstance(value, dict):
            nested_rows = extract_rows_from_payload(value)
            if nested_rows:
                return nested_rows

    return []


def normalise_rows(rows):
    normalised = []

    for row in rows:
        row = parse_json_if_needed(row)

        if isinstance(row, dict) and "data" in row:
            inner = parse_json_if_needed(row.get("data"))

            if isinstance(inner, dict):
                normalised.append(inner)
                continue

        if isinstance(row, dict):
            normalised.append(row)

    return normalised


def fetch_jobs():
    url = f"{API_URL}/jobs"
    params = {"all_records": "true", "limit": 5000}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=20)
    except requests.RequestException as e:
        st.error(f"API connection failed: {e}")
        return pd.DataFrame()

    if response.status_code == 401:
        st.error("API key rejected. Check WORKLOG_API_KEY in worklog.service.")
        return pd.DataFrame()

    if response.status_code != 200:
        st.error(f"API error: {response.status_code} - {response.text}")
        return pd.DataFrame()

    payload = response.json()
    rows = extract_rows_from_payload(payload)
    rows = normalise_rows(rows)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    if "work_date" not in df.columns and "data" in df.columns:
        parsed_rows = normalise_rows(df["data"].tolist())
        df = pd.DataFrame(parsed_rows)

    if "work_date" not in df.columns:
        st.error("API response does not include 'work_date'.")
        st.write("API payload preview:")
        st.json(payload)
        st.dataframe(df.head(10))
        return pd.DataFrame()

    required_text_cols = [
        "job_id",
        "category",
        "job_status",
        "waiting_time",
        "vehicle_description",
        "vehicle_reg",
        "collection_from",
        "delivery_to",
        "job_expenses",
        "auth_code",
        "comments",
        "paid_date",
        "job_outcome",
    ]

    for col in required_text_cols:
        df = ensure_column(df, col, "")

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    df["work_date"] = pd.to_datetime(df["work_date"], errors="coerce")
    df["paid_date"] = pd.to_datetime(df["paid_date"], errors="coerce")

    money_cols = [
        "amount",
        "waiting_amount",
        "expenses_amount",
        "add_pay",
        "waiting_hours",
    ]

    for col in money_cols:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["gross_total"] = df["amount"] + df["waiting_amount"] + df["add_pay"]
    df["net_total"] = df["gross_total"] - df["expenses_amount"]

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


def show_financial_summary(job_amount, waiting_amount, add_pay, expenses_amount, valid_expenses):
    total_to_pay = round(
        float(job_amount) + float(waiting_amount) + float(add_pay) - float(expenses_amount),
        2,
    )

    with st.container(border=True):
        st.markdown("### 📊 Financial Summary")

        st.metric("Total To Pay", f"£{total_to_pay:,.2f}")

        st.divider()

        st.write(f"**Job Amount:** £{float(job_amount):,.2f}")
        st.write(f"**Waiting Amount:** £{float(waiting_amount):,.2f}")
        st.write(f"**Add Pay:** £{float(add_pay):,.2f}")
        st.write(f"**Expenses:** £{float(expenses_amount):,.2f}")

        st.divider()

        if valid_expenses:
            st.markdown("#### Expense Breakdown")
            for row in valid_expenses:
                st.write(f"- {row['type']}: £{float(row['amount']):,.2f}")
        else:
            st.caption("No expenses added.")

    return total_to_pay


if "power_add_expense_rows" not in st.session_state:
    st.session_state.power_add_expense_rows = [{"type": "No expenses", "amount": 0.0}]

if "power_edit_expense_rows" not in st.session_state:
    st.session_state.power_edit_expense_rows = [{"type": "No expenses", "amount": 0.0}]

if "power_edit_expense_row_id" not in st.session_state:
    st.session_state.power_edit_expense_row_id = None


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
    history_df = daily[daily["report_date"] < selected_date].copy()

    def has_seen_postcode_before(row):
        collection = clean_postcode(row.get("collection_from"))
        delivery = clean_postcode(row.get("delivery_to"))

        if history_df.empty:
            return "No"

        history_collection = (
            history_df["collection_from"]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.upper()
        )

        history_delivery = (
            history_df["delivery_to"]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.upper()
        )

        found_collection = bool(collection) and (
            history_collection.str.contains(collection, na=False, regex=False).any()
            or history_delivery.str.contains(collection, na=False, regex=False).any()
        )

        found_delivery = bool(delivery) and (
            history_collection.str.contains(delivery, na=False, regex=False).any()
            or history_delivery.str.contains(delivery, na=False, regex=False).any()
        )

        return "Yes" if found_collection or found_delivery else "No"

    def has_driven_vehicle_before(row):
        vehicle = clean_lower(row.get("vehicle_description"))

        if not vehicle or history_df.empty:
            return "No"

        history_vehicles = (
            history_df["vehicle_description"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        return "Yes" if history_vehicles.str.contains(vehicle, na=False, regex=False).any() else "No"

    if day_df.empty:
        st.warning("No jobs found for this date.")
    else:
        day_df["postcode_before"] = day_df.apply(has_seen_postcode_before, axis=1)
        day_df["vehicle_before"] = day_df.apply(has_driven_vehicle_before, axis=1)

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
            "vehicle_before",
            "collection_from",
            "delivery_to",
            "postcode_before",
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
    st.subheader("Edit Job")

    search_text = st.text_input("Search by job ID, vehicle reg, postcode, or vehicle description")

    edit_df = df.copy()

    if search_text:
        search = search_text.strip().lower()
        edit_df = edit_df[
            edit_df.astype(str)
            .apply(lambda row: row.str.lower().str.contains(search, na=False, regex=False).any(), axis=1)
        ]

    display_cols = [
        "id",
        "work_date",
        "job_id",
        "category",
        "job_status",
        "amount",
        "waiting_amount",
        "expenses_amount",
        "net_total",
        "vehicle_reg",
        "collection_from",
        "delivery_to",
        "job_outcome",
    ]

    available_display_cols = [col for col in display_cols if col in edit_df.columns]
    st.dataframe(edit_df[available_display_cols], use_container_width=True)

    row_ids = edit_df["id"].dropna().astype(int).tolist()

    if row_ids:
        selected_id = st.selectbox("Select row ID to edit", row_ids)
        selected = edit_df[edit_df["id"] == selected_id].iloc[0]

        if st.session_state.power_edit_expense_row_id != selected_id:
            st.session_state.power_edit_expense_rows = parse_expenses_for_edit(
                selected.get("job_expenses"),
                selected.get("expenses_amount") or 0.0,
            )
            st.session_state.power_edit_expense_row_id = selected_id

        st.info(
            f"Editing Row #{selected_id} | "
            f"Job: {selected.get('job_id') or ''} | "
            f"Vehicle: {selected.get('vehicle_reg') or ''}"
        )

        left_col, right_col = st.columns([2.2, 1])

        with left_col:
            with st.container(border=True):
                st.markdown("### 📋 Job Information")

                col1, col2, col3 = st.columns(3)

                edit_work_date = col1.date_input(
                    "Work date",
                    value=selected["work_date"].date() if pd.notna(selected["work_date"]) else date.today(),
                    format="YYYY-MM-DD",
                    key=f"edit_work_date_{selected_id}",
                )

                edit_job_id = col2.text_input(
                    "Job ID",
                    value=str(selected.get("job_id") or ""),
                    key=f"edit_job_id_{selected_id}",
                )

                edit_category = col3.selectbox(
                    "Category",
                    CATEGORY_OPTIONS,
                    index=safe_select_index(CATEGORY_OPTIONS, selected.get("category")),
                    key=f"edit_category_{selected_id}",
                )

                col4, col5 = st.columns(2)

                edit_job_status = col4.selectbox(
                    "Job status",
                    JOB_STATUS_OPTIONS,
                    index=safe_select_index(JOB_STATUS_OPTIONS, selected.get("job_status")),
                    key=f"edit_job_status_{selected_id}",
                )

                edit_job_outcome = col5.selectbox(
                    "Job outcome",
                    JOB_OUTCOME_OPTIONS,
                    index=safe_select_index(JOB_OUTCOME_OPTIONS, selected.get("job_outcome")),
                    key=f"edit_job_outcome_{selected_id}",
                )

            with st.container(border=True):
                st.markdown("### 🚗 Vehicle Details")

                col1, col2 = st.columns(2)

                edit_vehicle_description = col1.text_input(
                    "Vehicle description",
                    value=str(selected.get("vehicle_description") or ""),
                    key=f"edit_vehicle_description_{selected_id}",
                )

                edit_vehicle_reg = col2.text_input(
                    "Vehicle reg",
                    value=str(selected.get("vehicle_reg") or ""),
                    key=f"edit_vehicle_reg_{selected_id}",
                )

            with st.container(border=True):
                st.markdown("### 📍 Journey")

                col1, col2 = st.columns(2)

                edit_collection_from = col1.text_input(
                    "Collection from",
                    value=str(selected.get("collection_from") or ""),
                    key=f"edit_collection_from_{selected_id}",
                )

                edit_delivery_to = col2.text_input(
                    "Delivery to",
                    value=str(selected.get("delivery_to") or ""),
                    key=f"edit_delivery_to_{selected_id}",
                )

            with st.container(border=True):
                st.markdown("### 💰 Job Pay & Expenses")

                edit_amount = st.number_input(
                    "Amount (£)",
                    min_value=0.0,
                    value=float(selected.get("amount") or 0),
                    step=0.01,
                    key=f"edit_amount_{selected_id}",
                )

                st.markdown("#### Expenses")

                for i, expense in enumerate(st.session_state.power_edit_expense_rows):
                    exp_col1, exp_col2, exp_col3 = st.columns([2, 1, 0.7])

                    expense_type = exp_col1.selectbox(
                        f"Expense Type {i + 1}",
                        EXPENSE_OPTIONS,
                        index=safe_select_index(EXPENSE_OPTIONS, expense.get("type")),
                        key=f"power_edit_expense_type_{selected_id}_{i}",
                    )

                    expense_amount = exp_col2.number_input(
                        f"Amount {i + 1} (£)",
                        min_value=0.0,
                        value=float(expense.get("amount") or 0.0),
                        step=0.5,
                        key=f"power_edit_expense_amount_{selected_id}_{i}",
                    )

                    st.session_state.power_edit_expense_rows[i] = {
                        "type": expense_type,
                        "amount": float(expense_amount),
                    }

                    with exp_col3:
                        st.write("")
                        st.write("")
                        if len(st.session_state.power_edit_expense_rows) > 1:
                            if st.button("Remove", key=f"power_edit_remove_expense_{selected_id}_{i}"):
                                st.session_state.power_edit_expense_rows.pop(i)
                                st.rerun()

                if st.button("➕ Add Another Expense", key=f"power_edit_add_expense_{selected_id}"):
                    st.session_state.power_edit_expense_rows.append(
                        {"type": "No expenses", "amount": 0.0}
                    )
                    st.rerun()

            edit_job_expenses, edit_expenses_amount, edit_valid_expenses = build_expenses_from_rows(
                st.session_state.power_edit_expense_rows
            )

            with st.container(border=True):
                st.markdown("### ⏱ Waiting Time")

                col1, col2, col3 = st.columns(3)

                edit_waiting_time = col1.text_input(
                    "Waiting time",
                    value=str(selected.get("waiting_time") or ""),
                    placeholder="e.g. 10-11 or 09:00-11:30",
                    key=f"edit_waiting_time_{selected_id}",
                )

                edit_waiting_hours = parse_wait_range_to_hours(edit_waiting_time)
                edit_waiting_amount = round(edit_waiting_hours * WAITING_RATE, 2)

                col2.metric("Waiting Hours", f"{edit_waiting_hours:.2f}")
                col3.metric("Waiting Amount", f"£{edit_waiting_amount:.2f}")

            with st.container(border=True):
                st.markdown("### 📝 Extra Details")

                col1, col2 = st.columns(2)

                edit_add_pay = col1.number_input(
                    "Additional pay (£)",
                    min_value=0.0,
                    value=float(selected.get("add_pay") or 0),
                    step=0.01,
                    key=f"edit_add_pay_{selected_id}",
                )

                current_paid_date = safe_date_value(selected.get("paid_date"))

                edit_paid_date = col2.date_input(
                    "Paid date",
                    value=current_paid_date,
                    format="YYYY-MM-DD",
                    key=f"edit_paid_date_{selected_id}",
                )

                edit_auth_code = st.text_input(
                    "Auth code",
                    value=str(selected.get("auth_code") or ""),
                    key=f"edit_auth_code_{selected_id}",
                )

                edit_comments = st.text_area(
                    "Comments",
                    value=str(selected.get("comments") or ""),
                    key=f"edit_comments_{selected_id}",
                )

        with right_col:
            show_financial_summary(
                edit_amount,
                edit_waiting_amount,
                edit_add_pay,
                edit_expenses_amount,
                edit_valid_expenses,
            )

            submitted = st.button(
                "💾 Update Job",
                type="primary",
                use_container_width=True,
                key=f"power_update_job_{selected_id}",
            )

        if submitted:
            payload = {
                "work_date": str(edit_work_date),
                "job_id": edit_job_id,
                "category": edit_category,
                "job_status": edit_job_status,
                "amount": float(edit_amount),
                "waiting_time": edit_waiting_time or None,
                "waiting_hours": float(edit_waiting_hours),
                "waiting_amount": float(edit_waiting_amount),
                "vehicle_description": edit_vehicle_description,
                "vehicle_reg": edit_vehicle_reg.upper().strip() if edit_vehicle_reg else None,
                "collection_from": edit_collection_from,
                "delivery_to": edit_delivery_to,
                "job_expenses": edit_job_expenses,
                "expenses_amount": float(edit_expenses_amount),
                "auth_code": edit_auth_code or None,
                "comments": edit_comments or None,
                "add_pay": float(edit_add_pay),
                "paid_date": str(edit_paid_date) if edit_paid_date else None,
                "job_outcome": edit_job_outcome,
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
    st.subheader("Add New Job")

    left_col, right_col = st.columns([2.2, 1])

    with left_col:
        with st.container(border=True):
            st.markdown("### 📋 Job Information")

            col1, col2, col3 = st.columns(3)

            add_work_date = col1.date_input(
                "Work date",
                value=date.today(),
                format="YYYY-MM-DD",
                key="power_add_work_date",
            )

            add_job_id = col2.text_input("Job ID", key="power_add_job_id")

            add_category = col3.selectbox(
                "Category",
                CATEGORY_OPTIONS,
                key="power_add_category",
            )

            col4, col5 = st.columns(2)

            add_job_status = col4.selectbox(
                "Job status",
                JOB_STATUS_OPTIONS,
                key="power_add_job_status",
            )

            add_job_outcome = col5.selectbox(
                "Job outcome",
                JOB_OUTCOME_OPTIONS,
                key="power_add_job_outcome",
            )

        with st.container(border=True):
            st.markdown("### 🚗 Vehicle Details")

            col1, col2 = st.columns(2)

            add_vehicle_description = col1.text_input(
                "Vehicle description",
                key="power_add_vehicle_description",
            )

            add_vehicle_reg = col2.text_input(
                "Vehicle reg",
                key="power_add_vehicle_reg",
            )

        with st.container(border=True):
            st.markdown("### 📍 Journey")

            col1, col2 = st.columns(2)

            add_collection_from = col1.text_input(
                "Collection from",
                key="power_add_collection_from",
            )

            add_delivery_to = col2.text_input(
                "Delivery to",
                key="power_add_delivery_to",
            )

        with st.container(border=True):
            st.markdown("### 💰 Job Pay & Expenses")

            add_amount = st.number_input(
                "Amount (£)",
                min_value=0.0,
                step=0.01,
                key="power_add_amount",
            )

            st.markdown("#### Expenses")

            for i, expense in enumerate(st.session_state.power_add_expense_rows):
                exp_col1, exp_col2, exp_col3 = st.columns([2, 1, 0.7])

                expense_type = exp_col1.selectbox(
                    f"Expense Type {i + 1}",
                    EXPENSE_OPTIONS,
                    index=safe_select_index(EXPENSE_OPTIONS, expense.get("type")),
                    key=f"power_add_expense_type_{i}",
                )

                expense_amount = exp_col2.number_input(
                    f"Amount {i + 1} (£)",
                    min_value=0.0,
                    value=float(expense.get("amount") or 0.0),
                    step=0.5,
                    key=f"power_add_expense_amount_{i}",
                )

                st.session_state.power_add_expense_rows[i] = {
                    "type": expense_type,
                    "amount": float(expense_amount),
                }

                with exp_col3:
                    st.write("")
                    st.write("")
                    if len(st.session_state.power_add_expense_rows) > 1:
                        if st.button("Remove", key=f"power_add_remove_expense_{i}"):
                            st.session_state.power_add_expense_rows.pop(i)
                            st.rerun()

            if st.button("➕ Add Another Expense", key="power_add_add_expense"):
                st.session_state.power_add_expense_rows.append(
                    {"type": "No expenses", "amount": 0.0}
                )
                st.rerun()

        add_job_expenses, add_expenses_amount, add_valid_expenses = build_expenses_from_rows(
            st.session_state.power_add_expense_rows
        )

        with st.container(border=True):
            st.markdown("### ⏱ Waiting Time")

            col1, col2, col3 = st.columns(3)

            add_waiting_time = col1.text_input(
                "Waiting time",
                placeholder="e.g. 10-11 or 09:00-11:30",
                key="power_add_waiting_time",
            )

            add_waiting_hours = parse_wait_range_to_hours(add_waiting_time)
            add_waiting_amount = round(add_waiting_hours * WAITING_RATE, 2)

            col2.metric("Waiting Hours", f"{add_waiting_hours:.2f}")
            col3.metric("Waiting Amount", f"£{add_waiting_amount:.2f}")

        with st.container(border=True):
            st.markdown("### 📝 Extra Details")

            col1, col2 = st.columns(2)

            add_add_pay = col1.number_input(
                "Additional pay (£)",
                min_value=0.0,
                step=0.01,
                key="power_add_add_pay",
            )

            add_paid_date = col2.date_input(
                "Paid date",
                value=None,
                format="YYYY-MM-DD",
                key="power_add_paid_date",
            )

            add_auth_code = st.text_input(
                "Auth code",
                key="power_add_auth_code",
            )

            add_comments = st.text_area(
                "Comments",
                key="power_add_comments",
            )

    with right_col:
        show_financial_summary(
            add_amount,
            add_waiting_amount,
            add_add_pay,
            add_expenses_amount,
            add_valid_expenses,
        )

        submitted = st.button(
            "💾 Add Job",
            type="primary",
            use_container_width=True,
            key="power_add_submit",
        )

    if submitted:
        missing_fields = []

        if not clean_text(add_job_id):
            missing_fields.append("Job ID")

        if not clean_text(add_vehicle_description):
            missing_fields.append("Vehicle description")

        if float(add_amount) <= 0:
            missing_fields.append("Amount")

        if missing_fields:
            st.error("Please complete these required fields: " + ", ".join(missing_fields))
        else:
            payload = {
                "work_date": str(add_work_date),
                "job_id": add_job_id,
                "category": add_category,
                "job_status": add_job_status,
                "amount": float(add_amount),
                "waiting_time": add_waiting_time or None,
                "waiting_hours": float(add_waiting_hours),
                "waiting_amount": float(add_waiting_amount),
                "vehicle_description": add_vehicle_description,
                "vehicle_reg": add_vehicle_reg.upper().strip() if add_vehicle_reg else None,
                "collection_from": add_collection_from,
                "delivery_to": add_delivery_to,
                "job_expenses": add_job_expenses,
                "expenses_amount": float(add_expenses_amount),
                "auth_code": add_auth_code or None,
                "comments": add_comments or None,
                "add_pay": float(add_add_pay),
                "paid_date": str(add_paid_date) if add_paid_date else None,
                "job_outcome": add_job_outcome,
            }

            res = api_post("/jobs", payload)

            if res.status_code in [200, 201]:
                st.success("Job added successfully.")
                st.session_state.power_add_expense_rows = [{"type": "No expenses", "amount": 0.0}]
                st.rerun()
            else:
                st.error(f"Add failed: {res.status_code} - {res.text}")


with tab6:
    st.subheader("Have I been to this postcode?")

    postcode = st.text_input("Enter postcode").strip().replace(" ", "").upper()

    if postcode:
        postcode_df = df.copy()

        postcode_df["collection_clean"] = (
            postcode_df["collection_from"]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.upper()
        )

        postcode_df["delivery_clean"] = (
            postcode_df["delivery_to"]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.upper()
        )

        matches = postcode_df[
            postcode_df["collection_clean"].str.contains(postcode, na=False, regex=False)
            | postcode_df["delivery_clean"].str.contains(postcode, na=False, regex=False)
        ].copy()

        if matches.empty:
            st.warning("No, you have not been to this postcode before.")
        else:
            matches = matches.sort_values("work_date", ascending=False)

            total_visits = len(matches)
            last_seen = matches["work_date"].max().date()
            vehicle_count = matches["vehicle_description"].nunique()

            st.success(f"Yes, you have been to this postcode {total_visits} time(s).")

            col1, col2, col3 = st.columns(3)

            col1.metric("Times visited", total_visits)
            col2.metric("Last seen", str(last_seen))
            col3.metric("Different vehicles", vehicle_count)

            vehicle_summary = (
                matches.groupby("vehicle_description")
                .agg(
                    times_driven=("id", "count"),
                    last_seen=("work_date", "max"),
                )
                .reset_index()
                .sort_values("times_driven", ascending=False)
            )

            vehicle_summary["last_seen"] = vehicle_summary["last_seen"].dt.date

            st.subheader("Vehicle history for this postcode")
            st.dataframe(vehicle_summary, use_container_width=True)

            st.subheader("Matching jobs")

            display_cols = [
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

            available_cols = [col for col in display_cols if col in matches.columns]

            st.dataframe(matches[available_cols], use_container_width=True)