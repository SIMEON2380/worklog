from datetime import date, timedelta
from typing import Optional, Tuple
import pandas as pd


def to_clean_date_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="object")

    s2 = s.copy()

    if pd.api.types.is_datetime64_any_dtype(s2):
        return s2.dt.date

    s_str = s2.astype("string").str.strip()
    out = pd.Series([pd.NaT] * len(s2), index=s2.index, dtype="object")

    # ISO yyyy-mm-dd
    iso = pd.to_datetime(s_str, format="%Y-%m-%d", errors="coerce")
    iso_mask = iso.notna()
    if iso_mask.any():
        out.loc[iso_mask] = iso[iso_mask].dt.date

    # numeric formats
    rem = ~iso_mask
    num = pd.to_numeric(s_str.where(rem, pd.NA), errors="coerce")
    num_mask = num.notna()
    if num_mask.any():
        n = num[num_mask]

        # yyyymmdd
        ymd_mask = (n >= 19000101) & (n <= 21001231)
        if ymd_mask.any():
            ymd_vals = n[ymd_mask].astype("int64").astype(str)
            ymd_dt = pd.to_datetime(ymd_vals, format="%Y%m%d", errors="coerce")
            out.loc[ymd_dt.index] = ymd_dt.dt.date

        rest = n[~ymd_mask]

        # unix ms
        ms_mask = rest >= 10_000_000_000
        if ms_mask.any():
            ms_dt = pd.to_datetime(rest[ms_mask], unit="ms", errors="coerce", utc=False)
            out.loc[ms_dt.index] = ms_dt.dt.date

        # unix seconds
        sec_rest = rest[~ms_mask]
        sec_mask = (sec_rest >= 1_000_000_000) & (sec_rest <= 4_000_000_000)
        if sec_mask.any():
            sec_dt = pd.to_datetime(sec_rest[sec_mask], unit="s", errors="coerce", utc=False)
            out.loc[sec_dt.index] = sec_dt.dt.date

        # excel serial (1900)
        excel_rest = sec_rest[~sec_mask]
        excel_mask = (excel_rest >= 20000) & (excel_rest <= 80000)
        if excel_mask.any():
            ex = excel_rest[excel_mask]
            ex_dt = pd.to_datetime(ex, unit="D", origin="1899-12-30", errors="coerce")
            out.loc[ex_dt.index] = ex_dt.dt.date

        # excel serial (1904)
        still_nat_idx = out[out.isna()].index.intersection(excel_rest.index)
        if len(still_nat_idx) > 0:
            ex2 = excel_rest.loc[still_nat_idx]
            ex2_dt = pd.to_datetime(ex2, unit="D", origin="1904-01-01", errors="coerce")
            out.loc[ex2_dt.index] = ex2_dt.dt.date

    # fallback parse
    rem2 = out.isna()
    if rem2.any():
        dt_rest = pd.to_datetime(s_str[rem2], errors="coerce", dayfirst=True)
        out.loc[dt_rest.index] = dt_rest.dt.date

    # sanity clamp
    min_ok = date(2000, 1, 1)
    max_ok = date(2100, 12, 31)

    def clamp(d):
        if pd.isna(d) or d is None:
            return pd.NaT
        if isinstance(d, date) and (d < min_ok or d > max_ok):
            return pd.NaT
        return d

    return out.apply(clamp)


def safe_date_bounds(dates: Optional[pd.Series]) -> Tuple[date, date]:
    today = date.today()
    if dates is None or len(dates) == 0:
        return today, today
    s = pd.Series(dates)
    s = to_clean_date_series(s).dropna()
    if s.empty:
        return today, today
    vals = s.tolist()
    return min(vals), max(vals)


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())