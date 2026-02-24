def list_jobs(search: str = "", limit: int = 500) -> pd.DataFrame:
    conn = get_conn()
    search = (search or "").strip()

    if search:
        like = f"%{search}%"
        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE job_number LIKE ?
               OR postcode LIKE ?
               OR vehicle_description LIKE ?
               OR customer_name LIKE ?
            ORDER BY job_number DESC
            LIMIT ?
            """,
            conn,
            params=(like, like, like, like, limit),
        )
    else:
        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            ORDER BY job_number DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )

    conn.close()
    return df