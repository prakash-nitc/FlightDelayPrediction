"""Project-wide configuration: paths, data scope and modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Data scope -------------------------------------------------------------
# BTS Reporting Carrier On-Time Performance, first half of 2023.
YEAR = 2023
MONTHS = [1, 2, 3, 4, 5, 6]
# Restrict to flights where both endpoints are among the busiest airports.
# Keeps the weather download small while still covering most US traffic.
TOP_N_AIRPORTS = 50

# --- Target -----------------------------------------------------------------
# Standard DOT definition: a flight is "delayed" if it arrives 15+ minutes late.
DELAY_THRESHOLD_MIN = 15
TARGET = "IS_DELAYED"

# --- Modelling --------------------------------------------------------------
RANDOM_STATE = 42
# Time-based split: train on Jan-Apr, validate on May, test on June.
TRAIN_MONTHS = [1, 2, 3, 4]
VALID_MONTHS = [5]
TEST_MONTHS = [6]

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
