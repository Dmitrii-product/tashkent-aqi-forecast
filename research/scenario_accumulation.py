"""Could the emission-cut scenario gain more on days when pollution has built up for several days?

Accumulation alone does not change the percentage: what has piled up over several stagnant days
was emitted by the same sources, so if each of them emits X% less every day, the pile is X% smaller
too (concentration is linear in emissions). The site's scenario already scales the measured,
accumulated concentration, so it already gives the larger cut in µg/m³ on such days.

What can change is the mix. Windblown dust from outside the city comes with wind; local smoke
(heating, traffic, industry) piles up when the air stands still. So on the dirtiest days the
local share is probably above the monthly mean, and the cut in percent is larger. The World Bank
gives monthly means only, so this is a sensitivity check under an assumption:

  A (site): every hour has its day's mix (monthly shares on the 15th, day by day in between).
  B: windblown dust stays at its monthly mean in µg/m³ (share × the calendar month's mean PM2.5,
     interpolated the same way); everything above it comes from the other sources in their
     monthly proportions. Same average cut over a month, more of it on dirty days, less on
     clean ones.

Result (2026-10-27, winters Nov–Feb 2018/19–2025/26, 882 days). Same overall cut (15.3% vs
15.5%), but B moves it to the dirty days: PM2.5 −18.0% on days with AQI > 200 and −17.0% at
151–200 (A: 15.3–15.4% everywhere), −11.4% on days ≤ 100. Days with AQI > 150: 211 measured,
142 under A, 134 under B. The forecast of 27.10 for 28.10–2.11 changes by at most 1 AQI (its
levels are close to the month's mean). Not modelled: the haze–boundary-layer feedback (haze
cools the ground and strengthens the inversion), which would add a little on the worst days.

    PYTHONPATH=. .venv/bin/python research/scenario_accumulation.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import ROOT, load_target
from aqf.scenario import CUT, SOURCES, _vec, scale, shares_on

_CUT = _vec(CUT)  # the moderate scenario, as when this study was run

DUST = SOURCES.index("dust")


def mid_month_interp(monthly: pd.Series):
    """Value on a day from per-calendar-month values placed on the 15th, linear in between."""
    def on(day):
        d = pd.Timestamp(day).normalize()
        a = d.replace(day=15) if d.day >= 15 else (d.replace(day=1) - pd.Timedelta(days=1)).replace(day=15)
        b = (a + pd.offsets.MonthBegin(1)).replace(day=15)
        w = (d - a) / (b - a)
        return (1 - w) * monthly[a.month] + w * monthly[b.month]
    return on


def scale_b(pm: pd.Series, clim_on) -> pd.Series:
    days = pm.index.normalize()
    out = np.empty(len(pm))
    cache = {}
    for i, (d, v) in enumerate(zip(days, pm.values)):
        if d not in cache:
            s = shares_on(d)
            other = np.delete(s, DUST) @ np.delete(_CUT, DUST) / (1 - s[DUST])  # cut of the non-dust mix
            cache[d] = (s[DUST] * clim_on(d), other)
        dust_abs, other = cache[d]
        dust = min(v, dust_abs)
        out[i] = v - _CUT[DUST] * dust - other * (v - dust)
    return pd.Series(out, index=pm.index)


def main() -> None:
    pm = load_target()[0]["pm"].dropna()
    clim = pm.groupby(pm.index.month).mean()  # calendar-month mean PM2.5, 2018–2026
    clim_on = mid_month_interp(clim)
    a0 = pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)
    pa, pb = scale(pm), scale_b(pm, clim_on)
    day = pd.DataFrame({"base": a0.resample("D").mean(), "A": pd.Series(aqi.pm_to_aqi(pa.values), index=pm.index).resample("D").mean(),
                        "B": pd.Series(aqi.pm_to_aqi(pb.values), index=pm.index).resample("D").mean(),
                        "pm": pm.resample("D").mean(), "pmA": pa.resample("D").mean(), "pmB": pb.resample("D").mean(),
                        "n": a0.resample("D").count()})
    day = day[(day["n"] >= 18) & day.index.month.isin([11, 12, 1, 2])]

    print("Winters Nov–Feb 2018/19–2025/26, by the day's measured AQI:")
    day["bin"] = pd.cut(day["base"], [0, 100, 150, 200, 500], labels=["≤100", "101–150", "151–200", ">200"])
    g = day.groupby("bin", observed=True)
    t = pd.DataFrame({"days": g.size(), "pm_cut_A_%": 100 * (1 - g["pmA"].sum() / g["pm"].sum()),
                      "pm_cut_B_%": 100 * (1 - g["pmB"].sum() / g["pm"].sum()),
                      "aqi_drop_A": (g["base"].mean() - g["A"].mean()), "aqi_drop_B": (g["base"].mean() - g["B"].mean())})
    print(t.round(1).to_string(), "\n")
    print("Overall PM2.5 cut, A vs B:", round(100 * (1 - day["pmA"].sum() / day["pm"].sum()), 1), round(100 * (1 - day["pmB"].sum() / day["pm"].sum()), 1))
    for c in ("base", "A", "B"):
        print(f"{c:4s} mean AQI {day[c].mean():5.1f}  days > 150: {(day[c] > 150).sum():4d}  days > 100: {(day[c] > 100).sum():4d}")

    # the current forecast, both ways
    js = json.loads((ROOT / "output" / "forecast.json").read_text())
    h = pd.DataFrame(js["hourly_7d"])
    p0 = pd.Series(aqi.aqi_to_pm(h["aqi"].values), index=pd.to_datetime(h["t"]))
    f = pd.DataFrame({"base": aqi.pm_to_aqi(p0.values), "A": aqi.pm_to_aqi(scale(p0).values),
                      "B": aqi.pm_to_aqi(scale_b(p0, clim_on).values)}, index=p0.index)
    f = f.groupby(f.index.normalize()).agg(["mean", "size"])
    f = f[f[("base", "size")] == 24].xs("mean", axis=1, level=1)
    print("\nForecast of", js["generated_at"][:16], "daily mean AQI:")
    print(f.round(0).to_string())


if __name__ == "__main__":
    main()
