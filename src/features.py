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


# --------------------------------------------------------------------------- #
# Weather features
# --------------------------------------------------------------------------- #
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "snowfall",
                "cloud_cover", "wind_speed_10m", "wind_gusts_10m"]


def weather_severity(code: pd.Series) -> pd.Series:
    """Collapse WMO weather codes into an ordinal severity scale.

    0 clear/cloudy, 1 drizzle, 2 rain, 3 heavy rain, 4 snow, 5 heavy snow, 6 thunderstorm.
    """
    bins = {0: 0, 1: 0, 2: 0, 3: 0, 45: 1, 48: 1, 51: 1, 53: 1, 55: 1, 61: 2, 63: 2, 65: 3,
            80: 2, 81: 2, 82: 3, 71: 4, 73: 4, 75: 5, 77: 4, 85: 4, 86: 5, 95: 6, 96: 6, 99: 6}
    return code.map(bins).fillna(0).astype(np.int8)


def prepare_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Hourly weather per airport plus short rolling windows (rain/snow building up)."""
    w = weather.sort_values(["airport", "time"]).copy()
    w["wx_severity"] = weather_severity(w["weather_code"])
    g = w.groupby("airport", sort=False)
    w["precip_3h"] = g["precipitation"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    w["snow_3h"] = g["snowfall"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    w["gust_max_3h"] = g["wind_gusts_10m"].transform(lambda s: s.rolling(3, min_periods=1).max())
    # Daily totals at the airport capture all-day disruption (ground stops, de-icing).
    w["date"] = w["time"].dt.normalize()
    w["precip_day"] = w.groupby(["airport", "date"])["precipitation"].transform("sum")
    w["snow_day"] = w.groupby(["airport", "date"])["snowfall"].transform("sum")
    w["severity_day_max"] = w.groupby(["airport", "date"])["wx_severity"].transform("max")
    return w.drop(columns=["weather_code", "date"])


def add_weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join origin weather at the departure hour and destination weather at the arrival hour."""
    w = prepare_weather(weather)
    cols = [c for c in w.columns if c not in ("airport", "time")]

    df = df.copy()
    df["_dep_ts"] = df["flight_date"] + pd.to_timedelta(df["dep_hour"], unit="h")
    # Red-eyes land the next calendar day.
    arr_date = df["flight_date"] + pd.to_timedelta((df["crs_arr_min"] < df["crs_dep_min"]).astype(int), unit="D")
    df["_arr_ts"] = arr_date + pd.to_timedelta(df["arr_hour"], unit="h")

    origin_w = w.rename(columns={"airport": "origin", "time": "_dep_ts", **{c: f"origin_{c}" for c in cols}})
    dest_w = w.rename(columns={"airport": "dest", "time": "_arr_ts", **{c: f"dest_{c}" for c in cols}})
    df = df.merge(origin_w, on=["origin", "_dep_ts"], how="left")
    df = df.merge(dest_w, on=["dest", "_arr_ts"], how="left")
    return df.drop(columns=["_dep_ts", "_arr_ts"])


# --------------------------------------------------------------------------- #
# Airport state: how delayed is the airport right now?
# --------------------------------------------------------------------------- #
def _lagged_hourly_mean(df, airport_col, hour_col, value_col, lags=(1, 2)):
    """Mean of `value_col` at the same airport/date over the previous `lags` hours.

    Only strictly earlier hours are used, mirroring the live airport status
    (e.g. FAA delay board) an operator would see before this departure.
    """
    hourly = (df.groupby([airport_col, "flight_date", hour_col])[value_col]
                .agg(["sum", "count"]).reset_index())
    total = pd.Series(0.0, index=df.index)
    count = pd.Series(0.0, index=df.index)
    for lag in lags:
        key = df[[airport_col, "flight_date"]].assign(**{hour_col: df["dep_hour"] - lag})
        m = key.merge(hourly, on=[airport_col, "flight_date", hour_col], how="left")
        total += m["sum"].fillna(0).values
        count += m["count"].fillna(0).values
    return (total / count.replace(0, np.nan)), count


def add_airport_state_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["origin_recent_dep_delay"], df["origin_recent_deps"] = _lagged_hourly_mean(
        df, "origin", "dep_hour", "dep_delay")
    # Arrivals into the *destination* over the 2h before we leave - a congested
    # destination often means ground-delay programs for inbound flights.
    df["dest_recent_arr_delay"], df["dest_recent_arrs"] = _lagged_hourly_mean(
        df.assign(_arr_h=df["arr_hour"]), "dest", "_arr_h", "arr_delay")
    df["origin_recent_delayed_share"] = _lagged_hourly_mean(
        df.assign(_late=(df["dep_delay"] >= 15).astype(float)), "origin", "dep_hour", "_late")[0]
    for c in ["origin_recent_dep_delay", "dest_recent_arr_delay", "origin_recent_delayed_share"]:
        df[c] = df[c].fillna(0)
    return df
