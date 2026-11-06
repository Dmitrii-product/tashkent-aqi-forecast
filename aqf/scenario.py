"""Emission-cut scenarios, on their own tab and never mixed into the forecast: how much cleaner the
air would be if the contributions of some sources fell. Two scenarios, both the owner's (2026-10-27):
moderate (heating -30%, dust from outside -5%, transport and heavy industry -10%) and strong (-50%,
-10%, -25%, -25%).

Source shares: World Bank (2024), "Air Quality Assessment for Tashkent and the Roadmap for Air
Quality Management Improvement in Uzbekistan", Figure 18 (p. 15): modelled contributions of each
source to the monthly mean PM2.5 in Tashkent in 2021 (dispersion model, monthly R² 0.93 against
monitoring). Percentages as printed; October and December add up to 102% and 104% (rounding in
the figure), so every month is normalised to 100%.

Method: a source's contribution in µg/m³ falls by the same fraction as its emissions, so PM2.5
falls by sum(share × cut). The report's shares are monthly means; they are taken to hold on the
15th of each month and change linearly, day by day, in between (owner's choice, 2026-10-27):
heating is switched on gradually, and month steps would make its share jump on the 1st (15% in
October to 38% in November). Every hour, forecast or measured, is scaled by its day's factor and
converted to US AQI; a day is the mean of its hourly AQI, as everywhere else.
The AQI scale bends (2.5 points per µg/m³ between 100 and 150, 0.7 above 150), so the same cut
in PM2.5 moves a dirty day less in AQI points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import aqi

REPORT = ("https://documents1.worldbank.org/curated/en/099061424075531584/pdf/"
          "P1771081ec9e93071181d3140144918d6cd.pdf")
SOURCES = ("heating", "dust", "transport", "heavy", "light", "urban_dust", "waste")
# % of the monthly mean PM2.5, in the order of SOURCES (residential/commercial heating, boundary/
# windblown dust, transport, heavy industry, light industry, urban dust, waste burning)
SHARES = {
    1: (39, 36, 12, 6, 6, 1, 0), 2: (41, 32, 13, 6, 7, 0, 1), 3: (18, 45, 17, 5, 7, 7, 1),
    4: (9, 48, 19, 6, 7, 9, 2), 5: (2, 49, 23, 5, 8, 12, 1), 6: (2, 53, 21, 5, 7, 11, 1),
    7: (3, 45, 23, 4, 9, 14, 3), 8: (2, 45, 25, 5, 9, 13, 1), 9: (3, 44, 25, 4, 9, 13, 2),
    10: (15, 44, 19, 4, 9, 10, 1), 11: (38, 26, 16, 7, 6, 6, 1), 12: (44, 26, 14, 11, 6, 3, 0),
}
SCENARIOS = {
    "moderate": {"heating": 0.30, "dust": 0.05, "transport": 0.10, "heavy": 0.10},
    "strong": {"heating": 0.50, "dust": 0.10, "transport": 0.25, "heavy": 0.25},
}
CUT = SCENARIOS["moderate"]


def _vec(cut: dict) -> np.ndarray:
    return np.array([cut.get(k, 0.0) for k in SOURCES])


def _norm(month: int) -> np.ndarray:
    s = np.array(SHARES[month], float)
    return s / s.sum()


def factor(month: int, cut: dict = CUT) -> float:
    """What is left of PM2.5 under the scenario with the report's shares for that month (the 15th)."""
    return 1 - float(_norm(month) @ _vec(cut))


def shares_on(day) -> np.ndarray:
    """Normalised shares on a day: the report's monthly values on the 15th, linear in between."""
    d = pd.Timestamp(day).normalize()
    a = d.replace(day=15) if d.day >= 15 else (d.replace(day=1) - pd.Timedelta(days=1)).replace(day=15)
    b = (a + pd.offsets.MonthBegin(1)).replace(day=15)
    w = (d - a) / (b - a)
    return (1 - w) * _norm(a.month) + w * _norm(b.month)


def factor_on(day, cut: dict = CUT) -> float:
    return 1 - float(shares_on(day) @ _vec(cut))


def scale(pm: pd.Series, cut: dict = CUT) -> pd.Series:
    days = pm.index.normalize()
    f = {d: factor_on(d, cut) for d in days.unique()}
    return pm * days.map(f).to_numpy()


def _aqi(pm: pd.Series) -> pd.Series:
    return pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)


def build(fc: dict, pm: pd.Series) -> dict:
    """Data for the tab, per scenario: the forecast's next full days as is and under it, and the
    measured past (by calendar month and by winter) recomputed the same way."""
    pm = pm.dropna()
    n = _aqi(pm).resample("D").count()
    measured = n[n >= 18].index  # as daily_aqi: a day needs 18 measured hours
    return {"report": REPORT, "hist_from": f"{measured.min():%Y-%m-%d}", "hist_to": f"{measured.max():%Y-%m-%d}",
            "scenarios": [{"id": k, **_one(fc, pm, cut)} for k, cut in SCENARIOS.items()]}


def _one(fc: dict, pm: pd.Series, cut: dict) -> dict:
    h = pd.DataFrame(fc["hourly_7d"])
    p0 = pd.Series(aqi.aqi_to_pm(h["aqi"].values), index=pd.to_datetime(h["t"]))
    f = pd.DataFrame({"pm": p0, "pm_s": scale(p0, cut), "aqi": _aqi(p0), "aqi_s": _aqi(scale(p0, cut))})
    days = []
    for d, g in f.groupby(f.index.normalize()):
        if len(g) == 24:
            days.append({"date": f"{d:%Y-%m-%d}", "pm": round(float(g["pm"].mean()), 1), "pm_s": round(float(g["pm_s"].mean()), 1),
                         "aqi": round(g["aqi"].mean()), "aqi_s": round(g["aqi_s"].mean()),
                         "heat": round(100 * float(shares_on(d)[0])), "cut": round(100 * (1 - factor_on(d, cut)), 1)})

    pm_s = scale(pm, cut)
    a0, a1 = _aqi(pm).resample("D"), _aqi(pm_s).resample("D")
    day = pd.DataFrame({"base": a0.mean(), "scen": a1.mean(), "n": a0.count()})
    day = day[day["n"] >= 18]  # as daily_aqi: a day needs 18 measured hours
    months = []
    for m in range(1, 13):
        g = day[day.index.month == m]
        months.append({"m": m, "shares": list(SHARES[m]), "cut": round(100 * (1 - factor(m, cut)), 1), "days": len(g),
                       "aqi": round(g["base"].mean()), "aqi_s": round(g["scen"].mean()),
                       "red": round(30 * float((g["base"] > 150).mean()), 1), "red_s": round(30 * float((g["scen"] > 150).mean()), 1)})
    win = day[day.index.month.isin([11, 12, 1, 2])].copy()
    win["w"] = np.where(win.index.month <= 2, win.index.year - 1, win.index.year)
    winters = []
    for y, g in win.groupby("w"):
        if len(g) >= 60:  # a winter with at least two months measured
            winters.append({"winter": f"{y}/{str(y + 1)[2:]}", "days": len(g),
                            "aqi": round(g["base"].mean()), "aqi_s": round(g["scen"].mean()),
                            **{k: int((g[c] > lim).sum()) for k, c, lim in
                               (("red", "base", 150), ("red_s", "scen", 150), ("over100", "base", 100), ("over100_s", "scen", 100))}})
    wh = pm.index.month.isin([11, 12, 1, 2])  # winter PM2.5 cut over the measured hours
    return {"cut": {k: round(100 * v) for k, v in cut.items()}, "win_pm_cut": round(100 * (1 - pm_s[wh].sum() / pm[wh].sum()), 1),
            "days": days, "months": months, "winters": winters}
