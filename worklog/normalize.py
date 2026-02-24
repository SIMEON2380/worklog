import re


def clean_job_number(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip()
    # keep digits + letters, remove spaces
    v = re.sub(r"\s+", "", v)
    return v


def clean_text(value: str) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clean_postcode(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip().upper()
    v = re.sub(r"\s+", " ", v)
    return v


def normalize_job_type(value: str) -> str:
    return clean_text(value)


def normalize_status(value: str) -> str:
    return clean_text(value)


def normalize_expense_type(value: str) -> str:
    return clean_text(value)