"""Two quick checks after the committee week (docs/checks/2026-10-27-committee-vs-forecast.md):
from 28.10 to 2.11 2026 the air stayed at 150–186 while our forecast kept expecting a front, and the
error breakdown (aqf.attribution) put most of the miss on the weather forecast (-15 AQI at 1 day,
-33 at 3 days, the same sign every day). Owner's request 2026-11-04: ideas to improve the forecast.

Both on the site's forecast (06:00 issue, daily mean of the day `lead` days on, backtest of model
version 2026-10-28 with archived forecast weather, 2024/25 – October 2026; research/peaks_study.py).

  1. persistence: in an episode (daily AQI > 150 the day before the issue) the model forecasts a
     drop of 30+ — would "as yesterday", or half of it, have been better?
  2. carry-over: shift the forecast by phi x yesterday's log miss of our own 1-day forecast (the
     part of the miss beyond the weather lasts about a day, research/peaks_study.py). phi chosen on
     2024/25, as for model changes.

    PYTHONPATH=. .venv/bin/python research/episode_checks.py

Result (2026-11-04):
  weather part (forecast-weather minus actual-weather prediction), 2024–2026: -2.5 / -5.5 AQI at
  1 day (other days / episodes), -3.5 / -4.4 at 3 days; the model's own bias with forecast weather
  -1.9…+0.9 — no systematic over-trust of promised fronts; the committee week was an exception.
  1. episode + forecast drop of 30+ (n = 44 at each lead): MAE model 27.1 / 24.2 (1 / 3 days),
     persistence 67.2 / 69.4, half-half 42.8 / 43.6; actual 133 / 124, model 134 / 131 — the drop
     usually came. Rejected.
  2. 2024/25 best at phi = 0 (lead 1: 18.24, phi 0.1 18.28, 0.3 18.90; lead 3: 19.90, 0.1 20.14);
     2025/26 -0.2 at phi 0.2, October 2026 (19 days) -2 at phi 0.3. Worse on the choice winter:
     rejected by the rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqf.data import load_target
from aqf.models import daily_aqi
from research.peaks_study import lead_daily


def main() -> None:
    A = daily_aqi(load_target()[0]["pm"])
    g = lead_daily(A)
    p = g.pivot_table(index=["D", "lead"], columns="weather", values="F").dropna().reset_index()
    p["A"] = A.reindex(p["D"]).values
    p["A1"] = A.reindex(p["D"] - pd.to_timedelta(p["lead"] + 1, unit="D")).values  # the day before the issue
    p = p.dropna()
    p["season"] = np.where(p["D"].dt.month >= 9, p["D"].dt.year, p["D"].dt.year - 1)
    p = p[p["season"] >= 2024]
    p["episode"] = p["A1"] > 150
    print("weather part and bias, 2024–2026")
    print(p.groupby(["lead", "episode"]).apply(lambda x: pd.Series({
        "n": len(x), "wx_part": (x["forecast"] - x["actual"]).mean(), "err_fc": (x["forecast"] - x["A"]).mean()})).round(1))
    for L in (1, 3):
        x = p[(p["lead"] == L) & p["episode"] & (p["forecast"] < p["A1"] - 30)]
        print(f"1. lead {L}: n={len(x)} MAE model {(x['forecast'] - x['A']).abs().mean():.1f} "
              f"persistence {(x['A1'] - x['A']).abs().mean():.1f} half {((x['forecast'] + x['A1']) / 2 - x['A']).abs().mean():.1f}")
    g = g[g["weather"] == "forecast"]
    f1 = g[g["lead"] == 1].set_index("D")["F"]
    for L in (1, 3):
        x = g[g["lead"] == L].copy()
        yday = x["D"] - pd.to_timedelta(L + 1, unit="D")  # the last full day before the 06:00 issue
        x["m"] = np.log(A.reindex(yday).values / f1.reindex(yday).values)
        x = x.dropna(subset=["m", "A", "F"])
        for y, s in x.groupby("season"):
            res = {phi: (np.exp(np.log(s["F"]) + phi * s["m"]) - s["A"]).abs().mean() for phi in (0, 0.1, 0.2, 0.3, 0.5)}
            print(f"2. lead {L} season {y} n={len(s)}: " + "  ".join(f"phi {k}: {v:.2f}" for k, v in res.items()))


if __name__ == "__main__":
    main()
