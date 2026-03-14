import re
import pandas as pd
from io import BytesIO
from pypdf import PdfReader


PAY_CODE_MAP = {
    "JOB": "job_amount",
    "INTRAV": "expenses",
    "WT/NC": "waiting_time",
    "RGNWGT": "regional_waiting",
    "ADDPAY": "addpay",
}


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from uploaded payslip PDF
    """
    reader = PdfReader(BytesIO(uploaded_file.read()))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def parse_payslip_lines(text):
    """
    Parse payslip lines into structured rows
    """

    pattern = re.compile(
        r"(\d{2}/\d{2}/\d{4})\s+\S+\s+(\d+)\s+(.+?)\s+(JOB|INTRAV|WT/NC|RGNWGT|ADDPAY)\s+(-?\d+\.\d+)"
    )

    rows = []

    for line in text.splitlines():

        match = pattern.search(line)

        if match:
            work_date = match.group(1)
            job_id = match.group(2)
            description = match.group(3)
            pay_code = match.group(4)
            amount = float(match.group(5))

            rows.append(
                {
                    "work_date": work_date,
                    "job_id": job_id,
                    "description": description,
                    "pay_code": pay_code,
                    "mapped_field": PAY_CODE_MAP.get(pay_code),
                    "amount": amount,
                }
            )

    return pd.DataFrame(rows)


def summarise_jobs(df):
    """
    Aggregate payments per job
    """

    if df.empty:
        return df

    grouped = []

    for job_id, group in df.groupby("job_id"):

        row = {
            "job_id": job_id,
            "job_amount": 0,
            "expenses": 0,
            "waiting_time": 0,
            "regional_waiting": 0,
            "addpay": 0,
            "total_paid": 0,
        }

        for _, r in group.iterrows():

            field = r["mapped_field"]
            amount = r["amount"]

            if field:
                row[field] += amount

            row["total_paid"] += amount

        grouped.append(row)

    return pd.DataFrame(grouped)


def parse_uploaded_payslip(uploaded_file):
    """
    Full pipeline
    """

    text = extract_text_from_pdf(uploaded_file)

    raw_df = parse_payslip_lines(text)

    summary_df = summarise_jobs(raw_df)

    return raw_df, summary_df