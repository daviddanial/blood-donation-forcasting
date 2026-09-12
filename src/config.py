"""
config.py
=========
Central configuration for the Ethiopian National Blood Supply Chain
Predictive Intelligence analysis pipeline.

Every "magic number" in the project lives here so that the whole pipeline
can be calibrated or re-tuned from one file. Values are grounded in the
factual constants quoted in the grant proposal (see ANALYSIS_PLAN.md):

  - Demand ~200,000 units/yr   [proposal refs 2,37]
  - Collection ~87,000 units/yr [proposal refs 2,37]
  - ABO/Rh donor-base proportions (Addis Ababa EBTBS study) [ref 24]
  - Platelet shelf life 5-7 days, WHO wastage threshold <5% [refs 3,40]
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the project root so the project is portable)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


# ---------------------------------------------------------------------------
# Study window (5 years of daily data = ~1,825 observation points, ref [1])
# ---------------------------------------------------------------------------
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"


# ---------------------------------------------------------------------------
# National calibration constants
# ---------------------------------------------------------------------------
NATIONAL_YEARLY_COLLECTION = 87_000   # units collected per year (refs 2,37)
NATIONAL_YEARLY_DEMAND = 200_000      # units demanded per year   (refs 2,37)


# ---------------------------------------------------------------------------
# Blood components and their shelf life (days) / processing share
# ---------------------------------------------------------------------------
COMPONENTS = ["whole_blood", "prbc", "platelets", "ffp"]
COMPONENT_SHELF_LIFE = {
    "whole_blood": 42,   # packed/whole red cells up to 42 days
    "prbc": 42,
    "platelets": 7,      # 5-7 day shelf life (ref [3]) - drives wastage
    "ffp": 365,
}
# Share of total collected component-units produced per component.
# NOTE: a single whole-blood unit can yield several components; this share
# vector is an approximation of the *units counted per component* and is
# documented as such (see ANALYSIS_PLAN.md).
COMPONENT_SHARE = {
    "whole_blood": 0.30,
    "prbc": 0.45,
    "platelets": 0.10,
    "ffp": 0.15,
}


# ---------------------------------------------------------------------------
# ABO/Rh phenotype proportions of the voluntary donor base (ref [24]).
# These match the cited table: O 44.65%, A 28.41%, B 21.24%, AB 5.71%,
# Rh+ 94.82%, Rh- 5.18%, O- 2.76%. The 8-group marginal vector is:
#   O+ 41.89, O- 2.76, A+ 27.47, A- 0.94, B+ 20.06, B- 1.18, AB+ 5.41, AB- 0.30
# ---------------------------------------------------------------------------
ABO_RH_SHARE = {
    "O+": 0.4189,
    "O-": 0.0276,   # universal donor, only 2.76% - "high-stakes envelope"
    "A+": 0.2747,
    "A-": 0.0094,
    "B+": 0.2006,
    "B-": 0.0118,
    "AB+": 0.0541,
    "AB-": 0.0030,
}
BLOOD_GROUPS = list(ABO_RH_SHARE.keys())


# ---------------------------------------------------------------------------
# Regions (administrative) with relative collection weight, urban flag and a
# conflict-affected flag (regions disabled by the Afar-Somali / Amhara-Tigray
# conflicts, refs 42,43). Weights are normalised to 1 in the generator.
# ---------------------------------------------------------------------------
REGIONS = [
    # (name,                        weight, urban, conflict, disruption_prob)
    ("Addis Ababa",                 0.22, True,  False, 0.000),
    ("Oromia",                      0.20, False, False, 0.010),
    ("Amhara",                      0.18, False, True,  0.030),
    ("Tigray",                      0.08, False, True,  0.050),
    ("SNNPR",                       0.10, False, False, 0.015),
    ("Sidama",                      0.04, False, False, 0.010),
    ("Somali",                      0.05, False, True,  0.045),
    ("Afar",                        0.02, False, True,  0.050),
    ("Benishangul-Gumuz",           0.02, False, False, 0.020),
    ("Gambela",                     0.02, False, False, 0.020),
    ("Harari",                      0.02, True,  False, 0.005),
    ("South West Ethiopia",         0.03, False, False, 0.015),
]


# ---------------------------------------------------------------------------
# Calendar-effect multipliers
# ---------------------------------------------------------------------------
LENT_COLLECTION_FACTOR = 0.80    # ~20% contraction during Great Lent (ref 5,6)
RAMADAN_COLLECTION_FACTOR = 0.78  # ~20-30% contraction during Ramadan (refs 7-10)
FASTING_DEMAND_FACTOR = 1.05     # clinical need stays constant / rises during fasts

DEMAND_WEEKDAY_FACTOR = [1.04, 1.04, 1.04, 1.04, 1.01, 0.96, 0.94]  # Mon..Sun
WEEKDAY_FACTOR = [1.15, 1.12, 1.15, 1.10, 1.05, 0.85, 0.70]  # Mon..Sun
HOLIDAY_COLLECTION_FACTOR = 0.70
HOLIDAY_DEMAND_FACTOR = 1.10     # emergencies continue on holidays

ANNUAL_SEASON_AMPLITUDE = 0.06   # ±6% seasonal swing
ANNUAL_SEASON_PHASE_DAYS = 0.0   # tune so the trough lands in the rainy season
TREND_GROWTH_PER_YEAR = 0.012    # +1.2%/yr collection growth
NOISE_SIGMA = 0.10               # multiplicative log-normal noise on daily volume

# Conflict disruption: when a window is active, collection is suppressed.
CONFLICT_COLLECTION_KEEP = 0.15   # drop to 15% of normal during disruption
CONFLICT_WINDOW_MIN, CONFLICT_WINDOW_MAX = 14, 45  # window length (days)
CONFLICT_INTERVAL_MIN, CONFLICT_INTERVAL_MAX = 90, 220  # gap between windows


# ---------------------------------------------------------------------------
# Model / evaluation protocol (proposal 3.4): 48 mo train / 6 mo val / 6 mo test
# ---------------------------------------------------------------------------
TRAIN_MONTHS = 48
VAL_MONTHS = 6
TEST_MONTHS = 6
FORECAST_HORIZONS = [7, 30, 90]

# Feature-engineering spans (proposal 3.3 / refs 4,13,17)
LAGS = [1, 2, 7, 14, 28]
ROLLING_WINDOWS = [7, 30]

RNG_SEED = 42
