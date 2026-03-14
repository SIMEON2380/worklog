import re
import pandas as pd
from io import BytesIO

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


PAY_CODE_MAP = {
    "JOB": "job_amount",
    "INTRAV": "expenses",
    "WT/NC": "waiting_time",
    "RGNWGT": "regional_waiting",
    "ADDPAY": "addpay",
}


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from uploaded payslip PDF.
    """
    uploaded_file.seek(0)
    reader = PdfReader(BytesIO(uploaded_file.read()))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def parse_payslip_lines(text):
    """
    Parse payslip text into:
    - jobs_df: rows with a job_id
    - other_df: rows without a job_id but with a recognised pay code / deduction
    """
    rows = []

    if not text or not str(text).strip():
        empty_cols = [
            "work_date",
            "job_id",
            "description",
            "pay_code",
            "mapped_field",
            "amount",
            "line_type",
        ]
        empty_df = pd.DataFrame(columns=empty_cols)
        return empty_df.copy(), empty_df.copy()

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Actual BCA extracted line format:
    # 7002057_JOBS 11887417 JOB 0.0004/02/2026 NN7 4HQ - NN5 4WJ
    job_pattern = re.compile(
        r"^\S+\s+(\d{7,8})\s+"
        r"(JOB|INTRAV|WT/NC|RGNWGT|ADDPAY)\s+"
        r"(-?\d+\.\d{2})(\d{2}/\d{2}/\d{4})\s+"
        r"(.+)$"
    )

    # Deductions / other lines:
    # Deduction -5.0013/03/2026 ADMINISTRATION CHARGE w/e 13/03/2026
    other_pattern = re.compile(
        r"^([A-Za-z][A-Za-z\s/-]*)\s+"
        r"(-?\d+\.\d{2})(\d{2}/\d{2}/\d{4})\s+"
        r"(.+)$"
    )

    for line in lines:
        match = job_pattern.match(line)
        if match:
            job_id = str(match.group(1)).strip()
            pay_code = match.group(2).strip()
            amount = float(match.group(3))
            work_date = pd.to_datetime(match.group(4), dayfirst=True, errors="coerce")
            description = match.group(5).strip()

            rows.append(
                {
                    "work_date": work_date,
                    "job_id": job_id,
                    "description": description,
                    "pay_code": pay_code,
                    "mapped_field": PAY_CODE_MAP.get(pay_code),
                    "amount": amount,
                    "line_type": "job",
                }
            )
            continue

        match = other_pattern.match(line)
        if match:
            label = match.group(1).strip()
            amount = float(match.group(2))
            work_date = pd.to_datetime(match.group(3), dayfirst=True, errors="coerce")
            description = match.group(4).strip()

            rows.append(
                {
                    "work_date": work_date,
                    "job_id": "",
                    "description": f"{label} - {description}",
                    "pay_code": "OTHER",
                    "mapped_field": None,
                    "amount": amount,
                    "line_type": "other",
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        empty_cols = [
            "work_date",
            "job_id",
            "description",
            "pay_code",
            "mapped_field",
            "amount",
            "line_type",
        ]
        empty_df = pd.DataFrame(columns=empty_cols)
        return empty_df.copy(), empty_df.copy()

    jobs_df = df[df["line_type"] == "job"].copy().reset_index(drop=True)
    other_df = df[df["line_type"] == "other"].copy().reset_index(drop=True)

    return jobs_df, other_df


def summarise_jobs(jobs_df):
    """
    Aggregate payslip payments per job_id.
    """
    if jobs_df.empty:
        return pd.DataFrame(
            columns=[
                "job_id",
                "work_date",
                "description",
                "job_amount",
                "expenses",
                "waiting_time",
                "regional_waiting",
                "addpay",
                "total_paid",
            ]
        )

    grouped = []

    for job_id, group in jobs_df.groupby("job_id", dropna=False):
        row = {
            "job_id": str(job_id).strip(),
            "work_date": group["work_date"].dropna().min(),
            "description": " | ".join(
                sorted(set(group["description"].dropna().astype(str).str.strip()))
            ),
            "job_amount": 0.0,
            "expenses": 0.0,
            "waiting_time": 0.0,
            "regional_waiting": 0.0,
            "addpay": 0.0,
            "total_paid": 0.0,
        }

        for _, r in group.iterrows():
            field = r["mapped_field"]
            amount = float(r["amount"])

            if field in row:
                row[field] += amount

            row["total_paid"] += amount

        grouped.append(row)

    summary_df = pd.DataFrame(grouped)

    money_cols = [
        "job_amount",
        "expenses",
        "waiting_time",
        "regional_waiting",
        "addpay",
        "total_paid",
    ]

    for col in money_cols:
        summary_df[col] = (
            pd.to_numeric(summary_df[col], errors="coerce")
            .fillna(0.0)
            .round(2)
        )

    summary_df = summary_df.sort_values(
        ["work_date", "job_id"], ascending=[True, True]
    ).reset_index(drop=True)

    return summary_df


def build_db_reconciliation(summary_df, db_df):
    """
    Compare payslip summary against database records.

    Notes:
    - INTRAV = expenses
    - WT/NC = waiting_time
    - RGNWGT = regional_waiting
    - regional_waiting is shown from payslip but not included in DB expected total
      because you do not currently store RGNWGT in the database.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    if db_df is None or db_df.empty:
        result = summary_df.copy()
        result["db_job_amount"] = 0.0
        result["db_expenses"] = 0.0
        result["db_waiting_time"] = 0.0
        result["db_expected_total"] = 0.0
        result["difference"] = result["total_paid"]
        result["status"] = "Missing in DB"
        return result

    db = db_df.copy()
    db["job_id"] = db["job_id"].astype(str).str.strip()

    if "amount" not in db.columns:
        db["amount"] = 0.0

    if "expenses_amount" not in db.columns:
        if "expenses" in db.columns:
            db["expenses_amount"] = pd.to_numeric(
                db["expenses"], errors="coerce"
            ).fillna(0.0)
        else:
            db["expenses_amount"] = 0.0

    if "waiting_time_amount" not in db.columns:
        db["waiting_time_amount"] = 0.0

    db["amount"] = pd.to_numeric(db["amount"], errors="coerce").fillna(0.0)
    db["expenses_amount"] = pd.to_numeric(
        db["expenses_amount"], errors="coerce"
    ).fillna(0.0)
    db["waiting_time_amount"] = pd.to_numeric(
        db["waiting_time_amount"], errors="coerce"
    ).fillna(0.0)

    db_grouped = (
        db.groupby("job_id", as_index=False)
        .agg(
            db_job_amount=("amount", "sum"),
            db_expenses=("expenses_amount", "sum"),
            db_waiting_time=("waiting_time_amount", "sum"),
        )
    )

    db_grouped["db_expected_total"] = (
        db_grouped["db_job_amount"]
        + db_grouped["db_expenses"]
        + db_grouped["db_waiting_time"]
    ).round(2)

    result = summary_df.merge(db_grouped, on="job_id", how="left")

    for col in ["db_job_amount", "db_expenses", "db_waiting_time", "db_expected_total"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0).round(2)

    result["difference"] = (result["total_paid"] - result["db_expected_total"]).round(2)

    def get_status(row):
        if row["db_expected_total"] == 0 and row["total_paid"] > 0:
            return "Missing in DB"
        if abs(row["difference"]) < 0.01:
            return "Matched"
        if row["difference"] > 0:
            return "Payslip higher"
        return "DB higher"

    result["status"] = result.apply(get_status, axis=1)

    return result.sort_values(
        ["status", "work_date", "job_id"], ascending=[True, True, True]
    ).reset_index(drop=True)


def parse_uploaded_payslip(uploaded_file):
    """
    Full pipeline.
    Returns:
    - jobs_df
    - other_df
    - summary_df
    """
    text = extract_text_from_pdf(uploaded_file)
    jobs_df, other_df = parse_payslip_lines(text)
    summary_df = summarise_jobs(jobs_df)
    return jobs_df, other_df, summary_df