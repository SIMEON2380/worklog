import os
import requests

API_URL = os.getenv("WORKLOG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("WORKLOG_API_KEY") or os.getenv("API_KEY")


def fetch_jobs(params=None):
    headers = {"x-api-key": API_KEY} if API_KEY else {}

    response = requests.get(
        f"{API_URL}/jobs",
        headers=headers,
        params=params,
        timeout=15,
    )
    response.raise_for_status()

    payload = response.json()

    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]

    if isinstance(payload, list):
        return payload

    raise ValueError("Unexpected API response format")