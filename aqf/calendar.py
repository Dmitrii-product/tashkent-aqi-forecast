"""Days off in Uzbekistan: Sundays plus public holidays (fixed dates, Navruz, both Eids).

Transferred working days are not modelled; the effect of day type on AQI is small
(within ±3–4% of neighbouring days), so the features are offered to the model, not forced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIXED = ["01-01", "01-02", "03-08", "03-21", "03-22", "05-09", "09-01", "10-01", "12-08", "12-31"]
EID = [  # Ramazon hayit, Qurbon hayit
    "2018-06-15", "2018-08-21", "2019-06-04", "2019-08-12", "2020-05-24", "2020-07-31",
    "2021-05-13", "2021-07-20", "2022-05-02", "2022-07-09", "2023-04-21", "2023-06-28",
    "2024-04-10", "2024-06-16", "2025-03-30", "2025-06-06", "2026-03-20", "2026-05-27",
    "2027-03-10", "2027-05-16",
]


def holidays(years=range(2018, 2028)) -> set:
    days = {pd.Timestamp(f"{y}-{md}").date() for y in years for md in FIXED}
    days |= {pd.Timestamp(d).date() for d in EID}
    return days


_HOL = holidays()


def offday_features(T: pd.DatetimeIndex) -> pd.DataFrame:
    """Per target time: public holiday, day off (Sunday or holiday), and whether the
    previous / next day is off (pre-holiday evenings and post-holiday mornings differ)."""
    T = pd.DatetimeIndex(T)
    d = T.normalize()
    hol = np.array([x in _HOL for x in d.date])
    off = hol | (T.dayofweek == 6)
    prev = np.array([x in _HOL for x in (d - pd.Timedelta(days=1)).date]) | ((d - pd.Timedelta(days=1)).dayofweek == 6)
    nxt = np.array([x in _HOL for x in (d + pd.Timedelta(days=1)).date]) | ((d + pd.Timedelta(days=1)).dayofweek == 6)
    return pd.DataFrame({"hol_T": hol.astype(float), "off_T": off.astype(float),
                         "off_prev_T": prev.astype(float), "off_next_T": nxt.astype(float)})
