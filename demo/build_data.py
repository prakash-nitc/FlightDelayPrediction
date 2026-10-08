"""Precompute model outputs for the static demo (one small JSON per date x origin airport).

    python -m demo.build_data        # after `make tune`

Writes demo/data/<date>/<airport>.json and demo/meta.json. The demo page (demo/index.html) is served
as a static Hugging Face Space: https://huggingface.co/spaces/PrakashOO7/flight-delay-prediction

For every June 2023 flight:
  * p      — the tuned model's delay probability with the flight's real inputs
  * grid   — probabilities for what-if scenarios: departure delay in GRID minutes,
             under (a) the real weather and (b) thunderstorms at both airports
"""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, TEST_MONTHS

OUT = Path(__file__).parent
DATA = OUT / "data"
GRID = [0, 10, 15, 20, 30, 45, 60]

bundle = joblib.load(MODELS_DIR / "xgboost_tuned_departure.joblib")
MODEL, FEATS, THR = bundle["model"], bundle["features"], bundle["threshold"]
df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
df = df[df["month"].isin(TEST_MONTHS)].reset_index(drop=True)
f64 = df[FEATS].select_dtypes("float64").columns
df[f64] = df[f64].astype(np.float32)


def predict(frame):
    return MODEL.predict_proba(frame[FEATS])[:, 1]


def storm(frame):
    f = frame.copy()
    for side in ("origin", "dest"):
        f[f"{side}_wx_severity"] = 6
        f[f"{side}_severity_day_max"] = 6
    f["origin_gust_max_3h"] = np.maximum(f["origin_gust_max_3h"], 60)
    f["origin_wind_gusts_10m"] = np.maximum(f["origin_wind_gusts_10m"], 60)
    return f


base_p = predict(df)
grid = np.zeros((len(df), 2, len(GRID)), dtype=np.float32)
for wi, wx in enumerate([lambda f: f, storm]):
    w = wx(df)
    for gi, d in enumerate(GRID):
        s = w.copy()
        s["dep_delay"] = np.float32(d)
        s["departed_late"] = np.int8(d >= 15)
        grid[:, wi, gi] = predict(s)
        print(f"weather {wi} dep_delay {d:>3}: mean P = {grid[:, wi, gi].mean():.3f}")

q = lambda x: np.rint(np.asarray(x) * 1000).astype(int)  # 3-decimal ints keep files small
if DATA.exists():
    shutil.rmtree(DATA)
df = df.assign(p=base_p, date=df["flight_date"].dt.strftime("%Y-%m-%d"))
df["gi"] = np.arange(len(df))
n_files = 0
for (date, origin), g in df.sort_values("crs_dep_time").groupby(["date", "origin"], observed=True):
    rows = []
    for r in g.itertuples():
        rows.append([str(r.carrier), int(r.flight_number), str(r.dest), int(r.crs_dep_time), int(round(r.dep_delay)),
                     int(round(r.arr_delay)), int(r.IS_DELAYED), int(q(r.p)), q(grid[r.gi]).ravel().tolist(),
                     int(round(r.prev_arr_delay)) if r.prev_leg_missing == 0 else None,
                     int(r.origin_wx_severity), int(r.dest_wx_severity)])
    path = DATA / date / f"{origin}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, separators=(",", ":")))
    n_files += 1

y = df["IS_DELAYED"].to_numpy()
pred = base_p >= THR
(OUT / "meta.json").write_text(json.dumps({
    "dates": sorted(df["date"].unique()), "airports": sorted(df["origin"].astype(str).unique()),
    "threshold": THR, "grid": GRID,
    "columns": ["carrier", "flight", "dest", "sched_dep", "dep_delay", "arr_delay", "actual", "p_x1000",
                "grid_x1000[weather(real,storm)][dep_delay]", "inbound_delay", "origin_wx", "dest_wx"],
    "check": {"accuracy": float((pred == (y == 1)).mean()), "n": int(len(y))},
}, indent=1))
print("files:", n_files, "accuracy check:", round(float((pred == (y == 1)).mean()), 4))
