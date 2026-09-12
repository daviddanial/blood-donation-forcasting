"""
calendar_features.py
====================
Engineering for the two big "moving-holiday" drivers in the proposal:

  1. Ethiopian Orthodox Great Lent ("Hudadi" / "Abiy Tsom") - a 55-56 day
     fast whose window moves EVERY year on the Gregorian calendar because it
     is anchored to Orthodox Easter (a Julian-calendar computus).
  2. Ramadan (and Eid al-Fitr / Eid al-Adha) - a lunar (Hijri) calendar
     that shifts ~11 days EARLIER each Gregorian year.

Plus standard temporal encodings (cyclic day-of-week / month-of-year) and a
hand-curated Ethiopian public-holiday list.

The module is dependency-light: Orthodox Easter + Gregorian<->Ethiopian dates
are computed here with pure-Python algorithms; only `convertdate` is used for
the Hijri (Islamic) calendar.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

try:
    from convertdate import islamic
    _HAS_CONVERTDATE = True
except Exception:  # pragma: no cover - optional
    _HAS_CONVERTDATE = False


# ---------------------------------------------------------------------------
# Orthodox Easter (Julian computus) -> Gregorian date
# ---------------------------------------------------------------------------
def julian_to_gregorian(y: int, m: int, d: int) -> date:
    """Convert a date in the Julian calendar to the Gregorian calendar."""
    offset = (y // 100) - (y // 400) - 2   # day difference (13 for 1900-2099)
    return date(y, m, d) + timedelta(days=offset)


def orthodox_easter(year: int) -> date:
    """Return the Gregorian date of Orthodox (Julian) Easter for `year`.

    Uses Meeus' algorithm to find Easter in the *Julian* calendar, then
    converts to the Gregorian calendar.
    """
    a = year % 4
    b = year % 7
    c = year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month = (d + e + 114) // 31
    day = ((d + e + 114) % 31) + 1
    return julian_to_gregorian(year, month, day)


def great_lent_window(year: int, length_days: int = 55) -> tuple[date, date]:
    """Return (start, end) Gregorian dates of the Great Lent fasting window.

    The proposal quotes a 55-56 day fast ("Abiy Tsom / Hudadi"). Here we flag
    the 55-day window that ends on Orthodox Easter.
    """
    easter = orthodox_easter(year)
    start = easter - timedelta(days=length_days)
    return start, easter


# ---------------------------------------------------------------------------
# Hijri calendar helpers (optional dependency `convertdate`)
# ---------------------------------------------------------------------------
def hijri_month_of(d: date) -> int | None:
    """Return the Hijri month (1-12) of `d`, or None if convertdate absent."""
    if not _HAS_CONVERTDATE:
        return None
    return islamic.from_gregorian(d.year, d.month, d.day)[1]


def is_ramadan(d: date) -> bool:
    return hijri_month_of(d) == 9


def is_eid(d: date) -> bool:
    """True on Eid al-Fitr (10/1) or Eid al-Adha (12/10)."""
    hij = islamic.from_gregorian(d.year, d.month, d.day) if _HAS_CONVERTDATE else None
    if hij is None:
        return False
    return (hij[1], hij[2]) in {(10, 1), (12, 10)}


def ethiopian_month_day(d: date) -> tuple[int, int] | None:
    """Return the (Ethiopian month, day) of `d` (Ethiopic calendar)."""
    try:
        from convertdate import ethiopian
        em, ed, _ = ethiopian.from_gregorian(d.year, d.month, d.day)
        return em, ed
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public holidays (approximate, curated Ethiopian list)
# ---------------------------------------------------------------------------
FIXED_HOLIDAYS = {
    (1, 7): "Ethiopian Christmas (Genna)",
    (1, 19): "Timkat (Epiphany)",
    (3, 2): "Adwa Victory Day",
    (5, 1): "International Workers' Day",
    (5, 5): "Patriots' Victory Day",
    (9, 11): "Enkutatash (Ethiopian New Year)",
    (9, 12): "Enkutatash (leap-year offset)",
    (9, 27): "Meskel (Finding of the True Cross)",
    (12, 28): "Christmas-related observance",
}


def is_public_holiday(d: date) -> bool:
    # Fixed Gregorian / Ethiopian-fixed holidays
    if (d.month, d.day) in FIXED_HOLIDAYS:
        return True
    # Movable religious feasts
    if is_eid(d):
        return True
    return False


# ---------------------------------------------------------------------------
# End-to-end frame builder
# ---------------------------------------------------------------------------
def build_calendar_frame(start: str, end: str) -> pd.DataFrame:
    """Build a per-day DataFrame of calendar features from `start` to `end`.

    Returns an index of `date` (DatetimeIndex) plus one column per feature.
    No look-ahead is introduced here; these are all "known-future" covariates.
    """
    dates = pd.date_range(start=start, end=end, freq="D")
    rows = []

    # Pre-compute Lent windows for every Gregorian year in range (+margin).
    year_min, year_max = dates.year.min(), dates.year.max()
    lent_map = {}
    for y in range(year_min - 1, year_max + 2):
        s, e = great_lent_window(y)
        lent_map[y] = (s, e)

    dow = dates.dayofweek.to_numpy()          # Mon=0 ... Sun=6
    month = dates.month.to_numpy()
    dom = dates.day.to_numpy()

    for i, dt in enumerate(dates):
        d = dt.date()
        lent_start, lent_end = lent_map[d.year]
        days_to_easter = (lent_end - d).days
        rows.append({
            "year": d.year,
            "month": d.month,
            "day_of_month": d.day,
            "day_of_week": int(dow[i]),           # 0=Mon .. 6=Sun
            "week_of_year": int(dt.isocalendar()[1]),
            "weekend": int(dow[i] >= 5),
            # Cyclic (Fourier) encodings so Jan 1 != Dec 31 for the model
            "dow_sin": float(np.sin(2 * np.pi * dow[i] / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow[i] / 7)),
            "month_sin": float(np.sin(2 * np.pi * (month[i] - 1) / 12)),
            "month_cos": float(np.cos(2 * np.pi * (month[i] - 1) / 12)),
            "is_lent": int(lent_start <= d <= lent_end),
            "days_until_easter": int(days_to_easter),
            "is_ramadan": int(is_ramadan(d)),
            "is_eid": int(is_eid(d)),
            "is_public_holiday": int(is_public_holiday(d)),
        })

    df = pd.DataFrame(rows, index=dates)
    df.index.name = "date"
    return df


if __name__ == "__main__":
    # Quick self-test
    cal = build_calendar_frame("2020-01-01", "2024-12-31")
    print(cal.head())
    print("Total days:", len(cal))
    print("Lent days:", int(cal.is_lent.sum()),
          "| Ramadan days:", int(cal.is_ramadan.sum()),
          "| Holidays:", int(cal.is_public_holiday.sum()))
    print("\nOrthodox Easter 2020-2024:")
    for y in (2020, 2021, 2022, 2023, 2024):
        print(" ", y, orthodox_easter(y))
