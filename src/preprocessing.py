"""Clean raw BTS flight records into a modelling-ready table.

Steps
-----
1. Keep flights where both origin and destination are in the top-N airports.
2. Drop cancelled / diverted flights (they have no arrival delay).
3. Drop rows missing the target or the aircraft tail number.
4. Remove exact duplicates and implausible values.
5. Convert hhmm clock times to minutes-after-midnight and build the target.
"""
import numpy as np
import pandas as pd

from src.config import DELAY_THRESHOLD_MIN, PROCESSED_DIR, TARGET
from src.data.airports import load_raw_flights, top_airports

RENAME = {
    "FlightDate": "flight_date",
    "Month": "month",
    "DayofMonth": "day_of_month",
    "DayOfWeek": "day_of_week",
    "Reporting_Airline": "carrier",
    "Tail_Number": "tail_number",
    "Flight_Number_Reporting_Airline": "flight_number",
    "Origin": "origin",
    "Dest": "dest",
    "OriginState": "origin_state",
    "DestState": "dest_state",
    "CRSDepTime": "crs_dep_time",
    "DepDelay": "dep_delay",
    "CRSArrTime": "crs_arr_time",
    "ArrDelay": "arr_delay",
    "CRSElapsedTime": "crs_elapsed_time",
    "Distance": "distance",
    "DistanceGroup": "distance_group",
}


def hhmm_to_minutes(hhmm: pd.Series) -> pd.Series:
    """1430 -> 870. BTS encodes midnight as 2400, which we map to 0."""
    hhmm = hhmm.astype(int) % 2400
    return (hhmm // 100) * 60 + hhmm % 100


def clean_flights(raw: pd.DataFrame, airports: list[str]) -> pd.DataFrame:
    n0 = len(raw)
    df = raw[raw["Origin"].isin(airports) & raw["Dest"].isin(airports)]
    n_scope = len(df)

    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]
    df = df.dropna(subset=["ArrDelay", "DepDelay", "Tail_Number", "CRSElapsedTime"])
    df = df.drop_duplicates()

    df = df[list(RENAME)].rename(columns=RENAME).copy()
    df["flight_date"] = pd.to_datetime(df["flight_date"])

    # Sanity filters: non-positive durations and extreme early arrivals are data errors.
    df = df[(df["crs_elapsed_time"] > 0) & (df["distance"] > 0)]
    df = df[df["arr_delay"] > -120]

    df["crs_dep_min"] = hhmm_to_minutes(df["crs_dep_time"])
    df["crs_arr_min"] = hhmm_to_minutes(df["crs_arr_time"])
    df["dep_hour"] = df["crs_dep_min"] // 60
    df["arr_hour"] = df["crs_arr_min"] // 60

    df[TARGET] = (df["arr_delay"] >= DELAY_THRESHOLD_MIN).astype(np.int8)

    print(f"raw rows               : {n0:>10,}")
    print(f"within top airports    : {n_scope:>10,}")
    print(f"after cleaning         : {len(df):>10,}")
    print(f"delay rate (>= {DELAY_THRESHOLD_MIN} min) : {df[TARGET].mean():>10.3f}")
    return df.reset_index(drop=True)


def main():
    raw = load_raw_flights()
    df = clean_flights(raw, top_airports())
    out = PROCESSED_DIR / "flights_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
