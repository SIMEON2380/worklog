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