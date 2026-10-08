"""Airport selection and coordinates.

Coordinates come from the public OurAirports dataset and are used to query
historical weather at each airport.
"""
import glob

import pandas as pd

from src.config import RAW_DIR, TOP_N_AIRPORTS

OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def load_raw_flights(columns=None) -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW_DIR / "flights_*.parquet")))
    if not files:
        raise FileNotFoundError("No flight files found. Run src.data.download_flights first.")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def top_airports(n: int = TOP_N_AIRPORTS) -> list[str]:
    """Busiest airports by total departures + arrivals in the raw data."""
    df = load_raw_flights(columns=["Origin", "Dest"])
    counts = pd.concat([df["Origin"], df["Dest"]]).value_counts()
    return counts.head(n).index.tolist()


def airport_coordinates(codes: list[str]) -> pd.DataFrame:
    out = RAW_DIR / "airports.csv"
    if not out.exists():
        pd.read_csv(OURAIRPORTS_URL).to_csv(out, index=False)
    ap = pd.read_csv(out, usecols=["iata_code", "name", "latitude_deg", "longitude_deg", "type"])
    ap = ap[ap["iata_code"].isin(codes)]
    # A handful of IATA codes are shared by heliports/closed fields; keep the airport.
    ap = ap.sort_values("type").drop_duplicates("iata_code", keep="first")
    missing = set(codes) - set(ap["iata_code"])
    if missing:
        raise ValueError(f"No coordinates for: {sorted(missing)}")
    return ap.rename(columns={"iata_code": "airport", "latitude_deg": "lat", "longitude_deg": "lon"})[
        ["airport", "name", "lat", "lon"]
    ].reset_index(drop=True)


if __name__ == "__main__":
    codes = top_airports()
    print(airport_coordinates(codes).to_string())
