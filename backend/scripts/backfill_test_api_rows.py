import os
import sqlite3


DB_PATH = os.getenv("WORKLOG_DB_PATH", "/var/lib/worklog/worklog_test.db")
WAITING_RATE = 7.50


def parse_wait_range_to_hours(value: str) -> float:
    if not value:
        return 0.0

    s = str(value).strip().replace(" ", "")
    if "-" not in s:
        return 0.0

    start, end = s.split("-", 1)

    def to_minutes(part: str) -> int:
        if ":" in part:
            h, m = part.split(":")
            return int(h) * 60 + int(m)
        return int(part) * 60

    try:
        start_m = to_minutes(start)
        end_m = to_minutes(end)
    except Exception:
        return 0.0

    if end_m <= start_m:
        return 0.0

    return round((end_m - start_m) / 60, 2)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id,
            waiting_time,
            waiting_hours,
            waiting_amount,
            vehicle_description,
            job_expenses
        FROM work_logs
    """)
    rows = cur.fetchall()

    updated_count = 0

    for row in rows:
        row_id = row["id"]

        waiting_time = row["waiting_time"]
        waiting_hours = row["waiting_hours"]
        waiting_amount = row["waiting_amount"]
        vehicle_description = row["vehicle_description"]
        job_expenses = row["job_expenses"]

        new_vehicle_description = (
            vehicle_description.strip().upper()
            if isinstance(vehicle_description, str) and vehicle_description.strip()
            else vehicle_description
        )

        new_job_expenses = (
            "no expenses"
            if job_expenses is None or str(job_expenses).strip() == ""
            else str(job_expenses).strip().lower()
        )

        new_waiting_hours = waiting_hours
        new_waiting_amount = waiting_amount

        if waiting_time and (waiting_hours is None or waiting_amount is None):
            parsed_hours = parse_wait_range_to_hours(waiting_time)
            new_waiting_hours = parsed_hours
            new_waiting_amount = round(parsed_hours * WAITING_RATE, 2)

        if (
            new_vehicle_description != vehicle_description
            or new_job_expenses != job_expenses
            or new_waiting_hours != waiting_hours
            or new_waiting_amount != waiting_amount
        ):
            cur.execute(
                """
                UPDATE work_logs
                SET
                    vehicle_description = ?,
                    job_expenses = ?,
                    waiting_hours = ?,
                    waiting_amount = ?
                WHERE id = ?
                """,
                (
                    new_vehicle_description,
                    new_job_expenses,
                    new_waiting_hours,
                    new_waiting_amount,
                    row_id,
                ),
            )
            updated_count += 1

    conn.commit()
    conn.close()

    print(f"Backfill complete. Updated {updated_count} row(s).")
    print(f"Database used: {DB_PATH}")


if __name__ == "__main__":
    main()