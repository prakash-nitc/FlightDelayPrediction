"""Download monthly BTS On-Time Performance files and store them as parquet.

Source: US DOT Bureau of Transportation Statistics, "Reporting Carrier On-Time
Performance (1987-present)". One zip (~25 MB) per month.

Usage:
    python -m src.data.download_flights
"""
import io
import zipfile

import pandas as pd
import requests

from src.config import MONTHS, RAW_DIR, YEAR

BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Only the columns we actually use; the raw file has ~110.
USECOLS = [
    "FlightDate", "Year", "Month", "DayofMonth", "DayOfWeek",
    "Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline",
    "Origin", "OriginCityName", "OriginState",
    "Dest", "DestCityName", "DestState",
    "CRSDepTime", "DepTime", "DepDelay",
    "CRSArrTime", "ArrTime", "ArrDelay", "ArrDel15",
    "Cancelled", "Diverted",
    "CRSElapsedTime", "Distance", "DistanceGroup",
]


def download_month(year: int, month: int, force: bool = False) -> pd.DataFrame:
    out = RAW_DIR / f"flights_{year}_{month:02d}.parquet"
    if out.exists() and not force:
        print(f"[skip] {out.name} already exists")
        return pd.read_parquet(out)

    url = BTS_URL.format(year=year, month=month)
    print(f"[get ] {url}")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh, usecols=USECOLS, low_memory=False)

    df.to_parquet(out, index=False)
    print(f"[save] {out.name}: {len(df):,} rows")
    return df


def main():
    for m in MONTHS:
        download_month(YEAR, m)


if __name__ == "__main__":
    main()
