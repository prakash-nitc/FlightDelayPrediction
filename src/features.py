"""Feature engineering: time, route, aircraft-rotation and weather features.

Every feature here is something that is known at (or just before) the
scheduled departure time, so it could be computed for a live prediction.
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


# --------------------------------------------------------------------------- #
# Time features
# --------------------------------------------------------------------------- #
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_weekend"] = (df["day_of_week"] >= 6).astype(np.int8)
    df["day_of_year"] = df["flight_date"].dt.dayofyear

    # Cyclical encodings so 23:00 and 00:00 are close to each other.
    df["dep_hour_sin"] = np.sin(2 * np.pi * df["crs_dep_min"] / 1440)
    df["dep_hour_cos"] = np.cos(2 * np.pi * df["crs_dep_min"] / 1440)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["part_of_day"] = pd.cut(
        df["dep_hour"], bins=[-1, 5, 11, 16, 20, 23],
        labels=["night", "morning", "afternoon", "evening", "late"],
    ).cat.codes.astype(np.int8)

    # Distance (in days) to the nearest US federal holiday - travel peaks around them.
    start = df["flight_date"].min() - pd.Timedelta(days=30)
    end = df["flight_date"].max() + pd.Timedelta(days=30)
    holidays = USFederalHolidayCalendar().holidays(start=start, end=end).values
    dates = df["flight_date"].values
    idx = np.searchsorted(holidays, dates)
    prev_h = holidays[np.clip(idx - 1, 0, len(holidays) - 1)]
    next_h = holidays[np.clip(idx, 0, len(holidays) - 1)]
    days_prev = np.abs((dates - prev_h).astype("timedelta64[D]").astype(int))
    days_next = np.abs((next_h - dates).astype("timedelta64[D]").astype(int))
    df["days_to_holiday"] = np.minimum(days_prev, days_next).clip(max=30)
    return df


# --------------------------------------------------------------------------- #
# Route & congestion features
# --------------------------------------------------------------------------- #
def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Route identity plus scheduled-traffic counts (known from the timetable)."""
    df = df.copy()
    df["route"] = df["origin"] + "_" + df["dest"]
    df["is_interstate"] = (df["origin_state"] != df["dest_state"]).astype(np.int8)
    # Scheduled block time vs. distance: padded schedules absorb small delays.
    df["sched_speed_mph"] = df["distance"] / (df["crs_elapsed_time"] / 60)

    # Scheduled departures out of the origin in the same local hour (airport congestion),
    # and scheduled arrivals into the destination in the arrival hour.
    df["origin_hourly_deps"] = df.groupby(["origin", "flight_date", "dep_hour"])["flight_number"].transform("size")
    df["dest_hourly_arrs"] = df.groupby(["dest", "flight_date", "arr_hour"])["flight_number"].transform("size")
    df["origin_daily_deps"] = df.groupby(["origin", "flight_date"])["flight_number"].transform("size")
    df["carrier_daily_flights"] = df.groupby(["carrier", "flight_date"])["flight_number"].transform("size")
    df["route_daily_flights"] = df.groupby(["route", "flight_date"])["flight_number"].transform("size")
    return df
