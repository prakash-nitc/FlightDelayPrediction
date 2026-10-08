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
    # A few records have implausibly short scheduled times; cap at a realistic airliner speed.
    df["sched_speed_mph"] = (df["distance"] / (df["crs_elapsed_time"] / 60)).clip(upper=650)

    # Scheduled departures out of the origin in the same local hour (airport congestion),
    # and scheduled arrivals into the destination in the arrival hour.
    df["origin_hourly_deps"] = df.groupby(["origin", "flight_date", "dep_hour"])["flight_number"].transform("size")
    df["dest_hourly_arrs"] = df.groupby(["dest", "flight_date", "arr_hour"])["flight_number"].transform("size")
    df["origin_daily_deps"] = df.groupby(["origin", "flight_date"])["flight_number"].transform("size")
    df["carrier_daily_flights"] = df.groupby(["carrier", "flight_date"])["flight_number"].transform("size")
    df["route_daily_flights"] = df.groupby(["route", "flight_date"])["flight_number"].transform("size")
    return df


# --------------------------------------------------------------------------- #
# Aircraft rotation ("previous flight delay") features
# --------------------------------------------------------------------------- #
ROTATION_KEYS = ["flight_date", "carrier", "flight_number", "origin", "tail_number"]


def build_rotation_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Previous-leg information for every aircraft, computed on the *full* raw data.

    Delays propagate through an aircraft's daily rotation: if the inbound aircraft
    is late, the next departure usually is too. The inbound leg's status (and its
    ETA) is known to the airline before the outbound flight departs, so these are
    legitimate pre-departure features. Computed before the top-airport filter so
    inbound legs from smaller airports are not lost.
    """
    r = raw[["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin",
             "Tail_Number", "CRSDepTime", "CRSArrTime", "DepDelay", "ArrDelay", "Cancelled"]].copy()
    r = r.dropna(subset=["Tail_Number"])
    r.columns = ["flight_date", "carrier", "flight_number", "origin", "tail_number",
                 "crs_dep_time", "crs_arr_time", "dep_delay", "arr_delay", "cancelled"]
    r["flight_date"] = pd.to_datetime(r["flight_date"])
    r["crs_dep_min"] = (r["crs_dep_time"] % 2400 // 100) * 60 + r["crs_dep_time"] % 100
    r["crs_arr_min"] = (r["crs_arr_time"] % 2400 // 100) * 60 + r["crs_arr_time"] % 100

    r = r.sort_values(["tail_number", "flight_date", "crs_dep_min"])
    g = r.groupby(["tail_number", "flight_date"], sort=False)

    out = r[ROTATION_KEYS].copy()
    out["leg_of_day"] = g.cumcount() + 1
    out["prev_dep_delay"] = g["dep_delay"].shift(1)
    out["prev_arr_delay"] = g["arr_delay"].shift(1)
    out["prev_cancelled"] = g["cancelled"].shift(1)
    prev_arr_min = g["crs_arr_min"].shift(1)
    turnaround = r["crs_dep_min"] - prev_arr_min
    # Overnight arrivals produce negative gaps; those aren't same-day turns.
    out["turnaround_min"] = turnaround.where(turnaround >= 0)
    # Projected buffer: scheduled ground time minus how late the inbound arrives.
    # Negative => the aircraft lands after this flight was due to leave.
    out["inbound_slack_min"] = out["turnaround_min"] - out["prev_arr_delay"].fillna(0)
    # Cumulative arrival delay of the aircraft so far today (before this leg).
    arr = r["arr_delay"].fillna(0)
    out["cum_prev_arr_delay"] = arr.groupby([r["tail_number"], r["flight_date"]], sort=False).cumsum() - arr

    out["is_first_leg"] = (out["leg_of_day"] == 1).astype(np.int8)
    out["prev_cancelled"] = out["prev_cancelled"].fillna(0).astype(np.int8)
    return out.drop_duplicates(ROTATION_KEYS)


def add_rotation_features(df: pd.DataFrame, rotation: pd.DataFrame) -> pd.DataFrame:
    rotation = rotation.astype({"flight_number": df["flight_number"].dtype})
    out = df.merge(rotation, on=ROTATION_KEYS, how="left", validate="many_to_one")
    # No inbound info (first leg, or inbound cancelled/diverted): flag it, treat as zero delay.
    out["prev_leg_missing"] = out["prev_arr_delay"].isna().astype(np.int8)
    for c in ["prev_dep_delay", "prev_arr_delay", "cum_prev_arr_delay"]:
        out[c] = out[c].fillna(0)
    out["turnaround_min"] = out["turnaround_min"].fillna(out["turnaround_min"].median())
    out["inbound_slack_min"] = out["inbound_slack_min"].fillna(out["turnaround_min"])
    return out
