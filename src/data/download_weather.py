"""Download hourly historical weather for each selected airport.

Uses the free Open-Meteo archive API (ERA5 reanalysis). Times are requested in
each airport's local timezone so they line up with BTS scheduled times, which
are also local.

Usage:
    python -m src.data.download_weather
"""
import calendar
import time

import pandas as pd
import requests

from src.config import MONTHS, RAW_DIR, YEAR
from src.data.airports import airport_coordinates, top_airports

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
]


def fetch_airport(airport: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
    }
    for attempt in range(5):
        resp = requests.get(ARCHIVE_URL, params=params, timeout=120)
        if resp.status_code == 429:  # rate limited
            time.sleep(30 * (attempt + 1))
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError(f"Gave up on {airport} after repeated rate limiting")

    hourly = pd.DataFrame(resp.json()["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly.insert(0, "airport", airport)
    return hourly


def main(force: bool = False):
    out = RAW_DIR / "weather_hourly.parquet"
    if out.exists() and not force:
        print(f"[skip] {out.name} already exists")
        return

    start = f"{YEAR}-{min(MONTHS):02d}-01"
    last_day = calendar.monthrange(YEAR, max(MONTHS))[1]
    end = f"{YEAR}-{max(MONTHS):02d}-{last_day}"

    coords = airport_coordinates(top_airports())
    frames = []
    for i, row in coords.iterrows():
        print(f"[{i + 1:>2}/{len(coords)}] {row.airport}")
        frames.append(fetch_airport(row.airport, row.lat, row.lon, start, end))
        time.sleep(1)  # be polite to the free API

    weather = pd.concat(frames, ignore_index=True)
    weather.to_parquet(out, index=False)
    print(f"[save] {out.name}: {len(weather):,} rows")


if __name__ == "__main__":
    main()
