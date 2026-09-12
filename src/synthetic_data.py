"""
synthetic_data.py
=================
A domain-aware SIMULATION of the Ethiopian blood supply chain. This is the
practical path to start building BEFORE the real BSIS/DHIS2 data is released
behind the ethics/partnership steps (proposal SOP 1).

What it produces (written to `data/`):
  collection_long.csv   : date, region, component, blood_group, units  (collection)
  issuance_long.csv     : date, region, component, blood_group, units  (demand)
  national_daily.csv    : date, collection, issuance                    (national totals)

The simulation is CALIBRATED to the factual constants in the proposal:
  - ~87,000 units collected/yr vs ~200,000 units demanded/yr  -> the supply gap
  - ABO/Rh donor proportions from the Addis Ababa EBTBS study
  - Great Lent and Ramadan collection contractions (20-30%)
  - Weekly rhythm, annual seasonality, public holidays, regional conflict
    disruptions (Afar/Somali/Tigray/Amhara), and reproducible noise.

Everything is seeded -> identical output on every run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .calendar_features import build_calendar_frame


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------
def _normalize(weights: list[float]) -> list[float]:
    s = float(sum(weights))
    return [w / s for w in weights]


def _poisson_draws(rng, lam: np.ndarray) -> np.ndarray:
    """Integer unit counts from a Poisson process (counts are non-negative ints)."""
    return rng.poisson(lam).astype(int)


# ---------------------------------------------------------------------------
# Per-region daily total series (collection and issuance/demand)
# ---------------------------------------------------------------------------
def _region_daily_total(cal: pd.DataFrame, region: dict, rng, kind: str) -> np.ndarray:
    """Return a per-day integer series for one region.

    kind = 'collection' -> `_total(baseline_yearly)`  with fasting HOLIDAY dips
    kind = 'issuance'   -> clinical demand, held HIGH during fasts/holidays.
    """
    dates = cal.index
    n = len(dates)

    if kind == "collection":
        baseline_yearly = config.NATIONAL_YEARLY_COLLECTION * region["weight"]
        weekday = np.array(config.WEEKDAY_FACTOR)[cal["day_of_week"].to_numpy()]
        holiday_f = np.where(cal["is_public_holiday"].to_numpy(),
                             config.HOLIDAY_COLLECTION_FACTOR, 1.0)
        lent_f = np.where(cal["is_lent"].to_numpy(),
                          config.LENT_COLLECTION_FACTOR, 1.0)
        ramadan_f = np.where(cal["is_ramadan"].to_numpy(),
                             config.RAMADAN_COLLECTION_FACTOR, 1.0)
        # Ramadan drive shift: evening post-Iftar drives partly offset weekday effect
        calendar_mult = weekday * holiday_f * lent_f * ramadan_f
    else:  # issuance / demand
        baseline_yearly = config.NATIONAL_YEARLY_DEMAND * region["weight"]
        weekday = np.array(config.DEMAND_WEEKDAY_FACTOR)[cal["day_of_week"].to_numpy()]
        holiday_f = np.where(cal["is_public_holiday"].to_numpy(),
                             config.HOLIDAY_DEMAND_FACTOR, 1.0)
        lent_f = np.where(cal["is_lent"].to_numpy(),
                          config.FASTING_DEMAND_FACTOR, 1.0)
        ramadan_f = np.where(cal["is_ramadan"].to_numpy(),
                             config.FASTING_DEMAND_FACTOR, 1.0)
        calendar_mult = weekday * holiday_f * lent_f * ramadan_f

    base_daily = baseline_yearly / 365.25

    # Secular trend over the 5-year window
    year_idx = dates.year.to_numpy() - dates.year.min()
    trend = (1.0 + config.TREND_GROWTH_PER_YEAR) ** year_idx

    # Annual seasonality (~±6%)
    doy = dates.dayofyear.to_numpy()
    season = 1.0 + config.ANNUAL_SEASON_AMPLITUDE * np.sin(
        2 * np.pi * (doy + config.ANNUAL_SEASON_PHASE_DAYS) / 365.0)

    lam = base_daily * trend * season * calendar_mult

    # Conflict disruption windows (suppress collection for conflict regions)
    if region["conflict"] and kind == "collection":
        lam = _apply_conflict_windows(lam, region["disruption_prob"], rng)

    # Multiplicative log-normal noise introduces realistic day-to-day variance.
    noise = np.exp(config.NOISE_SIGMA * rng.standard_normal(n))
    return _poisson_draws(rng, lam * noise)


def _apply_conflict_windows(lam: np.ndarray, prob: float, rng) -> np.ndarray:
    """Multiply `lam` by CONFLICT_COLLECTION_KEEP inside random disruption windows."""
    n = len(lam)
    out = lam.copy()
    i = 0
    while i < n:
        if rng.random() < prob:
            win = rng.integers(config.CONFLICT_WINDOW_MIN,
                               config.CONFLICT_WINDOW_MAX + 1)
            end = min(n, i + int(win))
            out[i:end] *= config.CONFLICT_COLLECTION_KEEP
            i = end + int(rng.integers(config.CONFLICT_INTERVAL_MIN,
                                       config.CONFLICT_INTERVAL_MAX + 1))
        else:
            i += 1
    return out


# ---------------------------------------------------------------------------
# Allocate a region's daily totals into component x blood-group cells
# ---------------------------------------------------------------------------
def _component_group_shares(rng) -> np.ndarray:
    """32-dimensional share vector = COMPONENT_SHARE x ABO_RH_SHARE (normalised)."""
    comp = np.array([config.COMPONENT_SHARE[c] for c in config.COMPONENTS])
    abo = np.array([config.ABO_RH_SHARE[g] for g in config.BLOOD_GROUPS])
    shares = np.outer(comp, abo).ravel()               # (4,8) -> (32,)
    shares = shares / shares.sum()
    # Small Dirichlet perturbation so no two days are identical
    perturb = rng.dirichlet(shares * 1e3)              # concentration ~ proportional
    return perturb


def _expand_long(dates, region_name: str, totals: np.ndarray, rng) -> pd.DataFrame:
    """Turn a per-day total series into a long component x blood-group table."""
    shares = _component_group_shares(rng)
    ncomp, nabo = len(config.COMPONENTS), len(config.BLOOD_GROUPS)
    n = len(dates)
    counts = np.vstack([
        rng.multinomial(int(totals[d]), shares) for d in range(n)
    ])                                               # (n_days, 32) - rows sum to each day's total

    df = pd.DataFrame({
        "date": np.repeat(dates, ncomp * nabo),
        "region": region_name,
        "component": np.tile(np.repeat(config.COMPONENTS, nabo), n),
        "blood_group": np.tile(config.BLOOD_GROUPS, n * ncomp),
        "units": counts.ravel(),
    })
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate(seed: int = config.RNG_SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cal = build_calendar_frame(config.START_DATE, config.END_DATE)
    dates = cal.index
    n = len(dates)

    regions = [{"name": r[0], "weight": r[1], "urban": r[2],
                "conflict": r[3], "disruption_prob": r[4]} for r in config.REGIONS]
    weights = _normalize([r["weight"] for r in regions])
    for r, w in zip(regions, weights):
        r["weight"] = w

    coll_parts, iss_parts = [], []
    national_coll = np.zeros(n)
    national_iss = np.zeros(n)

    for region in regions:
        coll = _region_daily_total(cal, region, rng, "collection")
        iss = _region_daily_total(cal, region, rng, "issuance")
        national_coll += coll
        national_iss += iss
        coll_parts.append(_expand_long(dates, region["name"], coll, rng))
        iss_parts.append(_expand_long(dates, region["name"], iss, rng))

    coll_long = pd.concat(coll_parts, ignore_index=True)
    iss_long = pd.concat(iss_parts, ignore_index=True)

    national_daily = pd.DataFrame({
        "collection": national_coll.astype(int),
        "issuance": national_iss.astype(int),
    }, index=dates)
    national_daily.index.name = "date"

    coll_long.to_csv(config.DATA_DIR / "collection_long.csv", index=False)
    iss_long.to_csv(config.DATA_DIR / "issuance_long.csv", index=False)
    national_daily.to_csv(config.DATA_DIR / "national_daily.csv")

    return {"collection_long": coll_long,
            "issuance_long": iss_long,
            "national_daily": national_daily}


if __name__ == "__main__":
    out = generate()
    for k, v in out.items():
        print(f"{k}: {v.shape} rows  |  units sum = {int(v['units'].sum()) if 'units' in v else int(v['collection'].sum())}")
    nd = out["national_daily"]
    print("\nNational totals over window:")
    print(f"  collection/day mean = {nd['collection'].mean():.1f}  ({nd['collection'].sum()/5:.0f}/yr)")
    print(f"  issuance/day   mean = {nd['issuance'].mean():.1f}  ({nd['issuance'].sum()/5:.0f}/yr)")
