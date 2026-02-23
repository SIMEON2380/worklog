import pandas as pd
from worklog.normalize import clean_job_number


def row_completeness_score(df: pd.DataFrame) -> pd.Series:
    def filled(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.strip().ne("")

    score = pd.Series(0, index=df.index, dtype="int64")
    score += filled(df.get("vehicle_reg", pd.Series("", index=df.index))).astype(int) * 3
    score += filled(df.get("vehicle_description", pd.Series("", index=df.index))).astype(int) * 2
    score += filled(df.get("collection_from", pd.Series("", index=df.index))).astype(int) * 2
    score += filled(df.get("delivery_to", pd.Series("", index=df.index))).astype(int) * 2
    score += filled(df.get("auth_code", pd.Series("", index=df.index))).astype(int) * 2
    score += filled(df.get("job_status", pd.Series("", index=df.index))).astype(int) * 1
    score += pd.to_numeric(df.get("amount", pd.Series(0, index=df.index)), errors="coerce").fillna(0).gt(0).astype(int) * 2
    score += pd.to_numeric(df.get("expenses_amount", pd.Series(0, index=df.index)), errors="coerce").fillna(0).gt(0).astype(int) * 1
    score += filled(df.get("waiting_time", pd.Series("", index=df.index))).astype(int) * 1
    score += filled(df.get("comments", pd.Series("", index=df.index))).astype(int) * 1
    return score


def dedup_for_reporting(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["work_date_str"] = d["work_date"].astype(str)
    d["job_id"] = d["job_id"].apply(clean_job_number)
    d["score"] = row_completeness_score(d)
    d = d.sort_values(["work_date_str", "job_id", "score", "id"], ascending=[True, True, False, True])
    d = d.drop_duplicates(["work_date_str", "job_id"], keep="first")
    return d.drop(columns=["work_date_str", "score"], errors="ignore")