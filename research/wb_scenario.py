"""How much cleaner would Tashkent's air be under an emission-cut scenario, against our forecast?

Source shares: World Bank, "Air Quality Assessment for Tashkent and the Roadmap for Air Quality
Management Improvement in Uzbekistan" (June 2024), Figure 18 (p. 15): modelled contributions of
each source to the monthly mean PM2.5 in Tashkent in 2021 (dispersion model, monthly R² 0.93
against monitoring). Percentages as printed; October and December add up to 102% and 104%
(rounding in the figure), so every month is normalised to 100%.

Scenario (owner's, 2026-10-27): a source's contribution in µg/m³ falls by the same fraction as
its emissions — residential/commercial heating −30%, boundary/windblown dust −5%, transport −10%,
heavy industry −10%. PM2.5 then falls by sum(share × cut) for the month; every hour of the
forecast is scaled by its month's factor and converted back to US AQI.

Assumptions: (1) contributions are linear in emissions (true for primary particles, roughly for
secondary ones); (2) every day of a month has that month's average mix. On the dirtiest winter
days heating's share is likely above the monthly mean, so the gain on those days is likely
understated; (3) the 2021 mix still holds.

Monthly steps here; the site (aqf/scenario.py) has since switched to shares placed on the 15th
and changed day by day in between (owner's choice), which gives −11…−12% for late October.

Result (2026-10-27, monthly steps). PM2.5 −15…−16% in November–February, −10% in March, −9% in October, ~−6%
in summer. Our forecast of 27.10: daily mean AQI −6…−7 for 28–31 October (149 → 142 on the
28th), −12…−13 on 1–2 November (November's mix has 38% heating against 15% in October). Measured
winters 2018/19–2025/26 (Nov–Feb, 882 days): mean AQI −9…−13 each winter; days > 150 from 211 to
140 (−34%), days > 100 from 589 to 484 (−18%); November–December ~10 unhealthy days a month
become ~6–7.

    PYTHONPATH=. .venv/bin/python research/wb_scenario.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import ROOT, load_target
from aqf.scenario import SHARES, factor  # the shares and the scenario now live in the package (site tab)


def scen_aqi(aqi_vals, months) -> np.ndarray:
    f = np.array([factor(m) for m in months])
    return aqi.pm_to_aqi(aqi.aqi_to_pm(np.asarray(aqi_vals, float)) * f)


def main() -> None:
    print("PM2.5 reduction by month:", {m: f"{100 * (1 - factor(m)):.1f}%" for m in SHARES}, "\n")

    js = json.loads((ROOT / "output" / "forecast.json").read_text())
    print(f"Forecast issued {js['generated_at'][:16]}")
    h = pd.DataFrame(js["hourly_7d"])
    h["t"] = pd.to_datetime(h["t"])
    h["scen"] = scen_aqi(h["aqi"], h["t"].dt.month)
    h["day"] = h["t"].dt.normalize()
    g = h.groupby("day").agg(n=("aqi", "size"), base=("aqi", "mean"), scen=("scen", "mean"),
                             base_max=("aqi", "max"), scen_max=("scen", "max"))
    g = g[g["n"] == 24]
    g["diff"] = g["scen"] - g["base"]
    print("Next days, daily mean and maximum hourly AQI:")
    print(g.drop(columns="n").round(0).to_string(), "\n")

    # the same scenario on the measured past winters: how many unhealthy days would remain
    pm = load_target()[0]["pm"]
    a0 = pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)
    a1 = pd.Series(aqi.pm_to_aqi(pm.values * np.array([factor(x) for x in pm.index.month])), index=pm.index)
    day = pd.DataFrame({"base": a0.resample("D").mean(), "scen": a1.resample("D").mean(),
                        "n": a0.resample("D").count()})
    day = day[day["n"] >= 18]
    # the 90-day outlook beyond day 7 is climatology, and daily means cannot be scaled in AQI (the
    # scale bends at 150), so months are shown on the measured hours of past years instead
    mon = day[day.index.month.isin([10, 11, 12, 1, 2, 3])]
    years = mon.groupby(mon.index.month).apply(lambda x: x.index.year.nunique())
    by_m = mon.groupby(mon.index.month).agg(days=("base", "size"), mean_base=("base", "mean"), mean_scen=("scen", "mean"),
                                            red_base=("base", lambda s: (s > 150).sum()), red_scen=("scen", lambda s: (s > 150).sum()))
    by_m["red_per_month_base"] = by_m["red_base"] / by_m["days"] * 30
    by_m["red_per_month_scen"] = by_m["red_scen"] / by_m["days"] * 30
    print("Measured 2018-2026 by calendar month: mean AQI, unhealthy days (> 150) per 30 days:")
    print(by_m.assign(years=years).round(1).to_string(), "\n")

    win = day[day.index.month.isin([11, 12, 1, 2])].copy()
    win["winter"] = [f"{y - 1 if mo <= 2 else y}/{str(y if mo <= 2 else y + 1)[2:]}" for y, mo in zip(win.index.year, win.index.month)]
    w = win.groupby("winter").agg(days=("base", "size"), mean_base=("base", "mean"), mean_scen=("scen", "mean"),
                                  red_base=("base", lambda s: (s > 150).sum()), red_scen=("scen", lambda s: (s > 150).sum()),
                                  over100_base=("base", lambda s: (s > 100).sum()), over100_scen=("scen", lambda s: (s > 100).sum()))
    w = w[w["days"] >= 60]
    print("Measured winters (Nov-Feb), days with daily mean AQI > 150 and > 100:")
    print(w.round(0).to_string())
    print(w[["days", "red_base", "red_scen", "over100_base", "over100_scen"]].sum().to_string())


if __name__ == "__main__":
    main()
