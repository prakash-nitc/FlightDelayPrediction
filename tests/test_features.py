import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_airport_state_features,
    add_rotation_features,
    add_time_features,
    add_weather_features,
    build_rotation_table,
    weather_severity,
)


def _raw_leg(flight_number, origin, dest, dep, arr, arr_delay, dep_delay=0.0, tail="N1", cancelled=0.0):
    return {
        "FlightDate": "2023-03-01", "Reporting_Airline": "WN", "Flight_Number_Reporting_Airline": flight_number,
        "Origin": origin, "Dest": dest, "Tail_Number": tail, "CRSDepTime": dep, "CRSArrTime": arr,
        "DepDelay": dep_delay, "ArrDelay": arr_delay, "Cancelled": cancelled,
    }


@pytest.fixture
def rotation():
    raw = pd.DataFrame([
        _raw_leg(1, "DAL", "HOU", 700, 800, arr_delay=25.0, dep_delay=30.0),
        _raw_leg(2, "HOU", "MSY", 840, 940, arr_delay=5.0),
        _raw_leg(3, "MSY", "ATL", 1030, 1230, arr_delay=0.0),
    ])
    return build_rotation_table(raw)


def test_rotation_previous_leg(rotation):
    r = rotation.set_index("flight_number")
    assert r.loc[1, "leg_of_day"] == 1 and np.isnan(r.loc[1, "prev_arr_delay"])
    assert r.loc[2, "prev_arr_delay"] == 25.0
    assert r.loc[2, "prev_dep_delay"] == 30.0
    assert r.loc[2, "turnaround_min"] == 40          # 08:40 - 08:00
    assert r.loc[2, "inbound_slack_min"] == 15       # 40 min turn - 25 min late
    assert r.loc[3, "cum_prev_arr_delay"] == 30.0    # 25 + 5


def test_rotation_merge_flags_first_leg(rotation):
    df = pd.DataFrame({
        "flight_date": pd.to_datetime(["2023-03-01"] * 2), "carrier": ["WN", "WN"],
        "flight_number": [1, 2], "origin": ["DAL", "HOU"], "tail_number": ["N1", "N1"],
    })
    out = add_rotation_features(df, rotation).set_index("flight_number")
    assert out.loc[1, "prev_leg_missing"] == 1 and out.loc[1, "prev_arr_delay"] == 0
    assert out.loc[2, "prev_leg_missing"] == 0 and out.loc[2, "prev_arr_delay"] == 25


def test_time_features_holiday_and_cyclical():
    df = pd.DataFrame({
        "flight_date": pd.to_datetime(["2023-05-29", "2023-05-31"]),  # Memorial Day, +2 days
        "day_of_week": [1, 3], "crs_dep_min": [0, 720], "dep_hour": [0, 12],
    })
    out = add_time_features(df)
    assert out["days_to_holiday"].tolist() == [0, 2]
    assert out.loc[0, "dep_hour_cos"] == pytest.approx(1.0)
    assert out.loc[1, "dep_hour_cos"] == pytest.approx(-1.0)


def test_weather_severity_mapping():
    codes = pd.Series([0, 3, 53, 63, 65, 73, 75, 95, 999])
    assert weather_severity(codes).tolist() == [0, 0, 1, 2, 3, 4, 5, 6, 0]


def test_weather_joins_origin_at_departure_and_dest_at_arrival():
    hours = pd.date_range("2023-03-01", periods=48, freq="h")
    weather = pd.concat([
        pd.DataFrame({"airport": ap, "time": hours, "temperature_2m": base + np.arange(48),
                      "relative_humidity_2m": 50, "precipitation": 0.0, "snowfall": 0.0,
                      "cloud_cover": 0, "wind_speed_10m": 5.0, "wind_gusts_10m": 10.0, "weather_code": 0})
        for ap, base in [("DEN", 0), ("LAX", 100)]
    ])
    # Red-eye: departs 23:00, arrives 01:00 next day.
    df = pd.DataFrame({
        "flight_date": pd.to_datetime(["2023-03-01"]), "origin": ["DEN"], "dest": ["LAX"],
        "dep_hour": [23], "arr_hour": [1], "crs_dep_min": [23 * 60], "crs_arr_min": [60],
    })
    out = add_weather_features(df, weather)
    assert out.loc[0, "origin_temperature_2m"] == 23
    assert out.loc[0, "dest_temperature_2m"] == 100 + 25


def test_airport_state_uses_only_earlier_hours():
    df = pd.DataFrame({
        "flight_date": pd.to_datetime(["2023-03-01"] * 4),
        "origin": ["ORD"] * 4, "dest": ["LGA"] * 4,
        "dep_hour": [8, 9, 10, 10], "arr_hour": [11, 12, 13, 13],
        "dep_delay": [10.0, 30.0, 500.0, 500.0], "arr_delay": [0.0] * 4,
    })
    out = add_airport_state_features(df)
    # Flights at 10:00 see the 08:00 and 09:00 departures only - not each other.
    assert out.loc[2, "origin_recent_dep_delay"] == pytest.approx(20.0)
    assert out.loc[0, "origin_recent_dep_delay"] == 0  # nothing earlier
