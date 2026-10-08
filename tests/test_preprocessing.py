import pandas as pd

from src.config import TARGET
from src.preprocessing import clean_flights, hhmm_to_minutes


def _raw(**overrides):
    row = {
        "FlightDate": "2023-01-05", "Year": 2023, "Month": 1, "DayofMonth": 5, "DayOfWeek": 4,
        "Reporting_Airline": "AA", "Tail_Number": "N123AA", "Flight_Number_Reporting_Airline": 100,
        "Origin": "ATL", "OriginCityName": "Atlanta, GA", "OriginState": "GA",
        "Dest": "ORD", "DestCityName": "Chicago, IL", "DestState": "IL",
        "CRSDepTime": 830, "DepTime": 845.0, "DepDelay": 15.0,
        "CRSArrTime": 1000, "ArrTime": 1020.0, "ArrDelay": 20.0, "ArrDel15": 1.0,
        "Cancelled": 0.0, "Diverted": 0.0,
        "CRSElapsedTime": 150.0, "Distance": 606.0, "DistanceGroup": 3,
    }
    row.update(overrides)
    return row


def test_hhmm_to_minutes_handles_midnight():
    s = pd.Series([0, 5, 830, 1430, 2359, 2400])
    assert hhmm_to_minutes(s).tolist() == [0, 5, 510, 870, 1439, 0]


def test_clean_flights_drops_cancelled_diverted_and_out_of_scope():
    raw = pd.DataFrame([
        _raw(),
        _raw(Flight_Number_Reporting_Airline=101, Cancelled=1.0, ArrDelay=None),
        _raw(Flight_Number_Reporting_Airline=102, Diverted=1.0),
        _raw(Flight_Number_Reporting_Airline=103, Dest="XYZ"),
        _raw(Flight_Number_Reporting_Airline=104, Tail_Number=None),
    ])
    out = clean_flights(raw, airports=["ATL", "ORD"])
    assert out["flight_number"].tolist() == [100]


def test_target_uses_15_minute_threshold():
    raw = pd.DataFrame([
        _raw(Flight_Number_Reporting_Airline=1, ArrDelay=14.0),
        _raw(Flight_Number_Reporting_Airline=2, ArrDelay=15.0),
        _raw(Flight_Number_Reporting_Airline=3, ArrDelay=-10.0),
    ])
    out = clean_flights(raw, airports=["ATL", "ORD"]).sort_values("flight_number")
    assert out[TARGET].tolist() == [0, 1, 0]


def test_time_columns_converted():
    out = clean_flights(pd.DataFrame([_raw()]), airports=["ATL", "ORD"])
    assert out.loc[0, "crs_dep_min"] == 510
    assert out.loc[0, "dep_hour"] == 8
    assert out.loc[0, "arr_hour"] == 10
