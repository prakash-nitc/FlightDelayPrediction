"""Score flights with the tuned model.

Look up a specific flight in the feature table and print its delay probability:

    python -m src.predict --date 2023-06-15 --carrier AA --flight 1234

Or score every flight on a date at one origin:

    python -m src.predict --date 2023-06-15 --origin ORD
"""
import argparse

import joblib
import pandas as pd

from src.config import MODELS_DIR, PREDICTION_POINT, PROCESSED_DIR, TARGET


def load_model(prediction_point: str = PREDICTION_POINT):
    bundle = joblib.load(MODELS_DIR / f"xgboost_tuned_{prediction_point}.joblib")
    return bundle["model"], bundle["features"], bundle["threshold"]


def predict(df: pd.DataFrame, prediction_point: str = PREDICTION_POINT) -> pd.DataFrame:
    model, features, threshold = load_model(prediction_point)
    proba = model.predict_proba(df[features])[:, 1]
    out = df[["flight_date", "carrier", "flight_number", "origin", "dest", "crs_dep_time"]].copy()
    out["delay_probability"] = proba.round(3)
    out["predicted_delayed"] = (proba >= threshold).astype(int)
    if TARGET in df:
        out["actual_delayed"] = df[TARGET].values
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--carrier")
    p.add_argument("--flight", type=int)
    p.add_argument("--origin")
    p.add_argument("--prediction-point", choices=["departure", "pre_departure"], default=PREDICTION_POINT)
    args = p.parse_args()

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet",
                         filters=[("flight_date", "==", pd.Timestamp(args.date))])
    if args.carrier:
        df = df[df["carrier"] == args.carrier]
    if args.flight:
        df = df[df["flight_number"] == args.flight]
    if args.origin:
        df = df[df["origin"] == args.origin]
    if df.empty:
        raise SystemExit("No matching flights in the feature table.")

    pd.set_option("display.width", 140)
    print(predict(df, args.prediction_point).sort_values("crs_dep_time").to_string(index=False))


if __name__ == "__main__":
    main()
