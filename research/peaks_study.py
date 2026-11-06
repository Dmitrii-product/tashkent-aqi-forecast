"""Pollution peaks: what drives them, and how far the 1-day and 3-day forecasts miss them.

The owner (2026-10-30): "let's analyse the causes of pollution peaks; look closely at past sharp
rises — how wrong was the forecast; it seems the 1-day and 3-day forecasts do not catch peaks"
(28.10: 179 against 149 / 142 forecast; 29.10: 162 against 133 / 99).

Daily forecasts are rebuilt from the walk-forward backtest (data/processed/bt_hourly.parquet,
model version 2026-10-28) the way the site makes them: the 06:00 forecast, the mean of its hourly
AQI over the target day, 1 or 3 days after the issue day, against the measured daily AQI. Two
weather variants for the same forecasts: archived weather forecasts (as in real use; 2024/25,
2025/26, October 2026) and the actual weather (all seven winters 2020–2026; the upper bound).
October–March only. "Sharp rise" = daily AQI up 30+ on the day before; "red" = AQI above 150.

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
        PYTHONPATH=. .venv/bin/python research/peaks_study.py          # ~1 min

Writes data/processed/peaks.json.

Result (2026-10-30):
  * The forecast lags sharp changes. Archived weather, 1 day ahead: after a rise of 50+ AQI in a
    day it was 35 too low on average (30–50: 15 too low), after a fall of 30+ it was 16 too high;
    3 days ahead 27 / 10 / 17. Red days caught by the forecast number (> 150): 64% (1 day), 69%
    (3 days), with 27–33% false alarms; persistence is far worse (MAE 37 vs 20).
  * It is calibrated, not squashed: given the forecast, the actual is right on average (slope of
    log actual on log forecast 1.01; forecast 160+ -> actual 184). So no stretching fixes it: the
    misses are days the model did not see coming at all.
  * Weather-forecast error is the smaller part: on the same days, with the actual weather the
    under-forecast of red days 1 day ahead shrinks from 25 to 20 (weather ~20% of it; 3 days
    ahead ~30%). 29.10 is the exception: three days before, the weather models promised a weak
    front (6.6 km/h, 1 calm hour, rain, no inversion) that did not come (forecast 99, actual 162).
  * Missed red days are episode starts in moderate weather (wind 4.5 vs 3.7 km/h, 15 vs 20 calm
    hours, night inversion 1.8 vs 4.6 C, level the day before 109 vs 159), mostly October and
    January–February; caught ones are deep stagnation in November–December.
  * What the model misses with the right weather is hardly linked to anything we measure (all
    |rho| <= 0.13: wind, calm, inversion 850/925, humidity, snow, NO2, weekday) except yesterday's
    miss (rho 0.32: extra pollution persists about a day) and long dry spells (> 14 days: 12 too
    low, 41 days) - "days since rain" was already tested as an input and not chosen.
  * Causes of peaks overall: red days are 64% deep-stagnation days (16+ calm hours or a night
    inversion of 4 C at 850 hPa; 31% of all cold days) and 54% cold days (< 3 C, heating); with
    both, a day is red in 54% of cases, with neither in 8%. Red days come in November–January
    (187 of 226 since 2018). Rise days: weaker wind (z -0.5), more calm (+0.5), wind dropping from the day
    before, higher pressure (+0.3), colder (-0.35), no rain.
  * The red-day chance computed from the same forecast is the better peak signal: 88% of red days
    had a chance of 30%+ one day ahead (54% of such days were red; Brier skill 0.45, 3 days 0.38).
    On the site it is shown only for days 8–90 and in a tooltip.
  * Hourly peaks: the forecast's highest hour of tomorrow is 189 on days whose highest hour was
    over 200 (actual 256).
  * Issuing tomorrow's forecast later helps overall, not with sharp rises (same days, archived
    weather): MAE 19.6 from 06:00, 17.4 from 18:00, 16.0 from 21:00; red days caught 64 -> 72%;
    after a 50+ rise still -30 (checked separately in the session, not part of this script).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aqf import aqi
from aqf.data import PROC, load_target, load_weather
from aqf.models import daily_aqi
from aqf.satellite import no2_series

COLD = [10, 11, 12, 1, 2, 3]
LEADS = (1, 3)
SEASONS = {"forecast": 2024, "actual": 2020}  # first season of each weather variant used here
RISE, RED = 30, 150.5


def lead_daily(A: pd.Series) -> pd.DataFrame:
    """The 06:00 forecast for the day `lead` days after the issue day, as the site makes it."""
    hb = pd.read_parquet(PROC / "bt_hourly.parquet", columns=["origin", "pred_aqi", "T", "weather", "season"])
    o = hb[hb["origin"].dt.hour == 6].copy()
    o["D"] = o["T"].dt.normalize()
    o["lead"] = (o["D"] - o["origin"].dt.normalize()).dt.days
    o = o[o["lead"].isin(LEADS)]
    g = o.groupby(["weather", "lead", "D", "season"])["pred_aqi"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] == 24].drop(columns="size").rename(columns={"mean": "F"})
    g["A"], g["A1"] = A.reindex(g["D"]).values, A.reindex(g["D"] - pd.Timedelta(days=1)).values
    g["Aknown"] = [A.get(d - pd.Timedelta(days=lead + 1), np.nan) for d, lead in zip(g["D"], g["lead"])]  # last full day at issue
    g = g.dropna(subset=["A", "A1", "Aknown"])
    g = g[g["D"].dt.month.isin(COLD)]
    g = g[g["season"] >= g["weather"].map(SEASONS)]
    g["e"], g["dA"] = g["F"] - g["A"], g["A"] - g["A1"]
    return g


def daily_weather(A: pd.Series) -> pd.DataFrame:
    """Day by day: the actual weather (archive) and what it did since the day before."""
    w = load_weather().asfreq("h").interpolate(limit=3)
    ws, night = w["wind_speed_10m"], w.index.hour.isin(range(0, 9))
    day = lambda s, how="mean": getattr(s.resample("D"), how)()  # noqa: E731
    d = pd.DataFrame({
        "ws": day(ws), "calm": day((ws < 5).astype(float), "sum"), "gust": day(w["wind_gusts_10m"], "max"),
        "t": day(w["temperature_2m"]), "p": day(w["surface_pressure"]), "rain": day(w["precipitation"], "sum"),
        "rh": day(w["relative_humidity_2m"]), "cloud": day(w["cloud_cover"]), "ws850": day(w["wind_speed_850hPa"]),
        "inv": day((w["temperature_850hPa"] - w["temperature_2m"]).where(night), "max"),  # night inversion, C
        "inv925": day((w["temperature_925hPa"] - w["temperature_2m"]).where(night), "max"),
        "snow3": day(w["snowfall"], "sum").rolling(3).sum(),
    })
    d["dt"], d["dp"], d["dws"] = d["t"].diff(), d["p"].diff(), d["ws"].diff()
    wet = d["rain"] >= 1
    d["dry"] = (~wet).astype(int).groupby(wet.cumsum()).cumsum().shift(1).clip(upper=40)  # dry days before
    stag = d["ws"] < d["ws"].groupby(d.index.month).transform("median")
    d["stag_run"] = stag.astype(int).groupby((~stag).cumsum()).cumsum()  # calm-ish days in a row
    no2 = no2_series()["no2"]
    d["no2"] = no2[~no2.index.duplicated(keep="last")].reindex(d.index)
    d["dow"] = d.index.dayofweek
    d["A"], d["dA"] = A, A.diff()
    return d


def _z(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Anomaly against the same calendar month, in standard deviations."""
    g = d[cols].groupby(d.index.month)
    return (d[cols] - g.transform("mean")) / g.transform("std")


def errors_by_change(g: pd.DataFrame) -> dict:
    out = {}
    for (w, L), x in g.groupby(["weather", "lead"]):
        t = x.groupby(pd.cut(x["dA"], [-999, -30, -10, 10, 30, 50, 999]), observed=True).agg(
            n=("e", "size"), bias=("e", "mean"), mae=("e", lambda v: v.abs().mean()), actual=("A", "mean"), forecast=("F", "mean"))
        out[f"{w}_L{L}"] = {"mae": round(float(x["e"].abs().mean()), 1),
                            "persistence_mae": round(float((x["Aknown"] - x["A"]).abs().mean()), 1),
                            "by_change": [{"change": str(k), **{c: round(float(v), 1) for c, v in r.items()}} for k, r in t.iterrows()]}
    return out


def calibration(g: pd.DataFrame) -> dict:
    out = {}
    for (w, L), x in g.groupby(["weather", "lead"]):
        t = x.groupby(pd.cut(x["F"], [0, 80, 100, 120, 140, 160, 999]), observed=True).agg(n=("A", "size"), forecast=("F", "mean"), actual=("A", "mean"))
        red, hit = x["A"] > RED, x["F"] > RED
        out[f"{w}_L{L}"] = {
            "slope_log_actual_on_forecast": round(float(np.polyfit(np.log(x["F"]), np.log(x["A"]), 1)[0]), 2),
            "share_of_change_followed": round(float(np.polyfit(x["A"] - x["Aknown"], x["F"] - x["Aknown"], 1)[0]), 2),
            "red_days": int(red.sum()), "red_caught": round(float((hit & red).sum() / red.sum()), 2),
            "false_alarms": round(float((hit & ~red).sum() / max(hit.sum(), 1)), 2),
            "by_forecast": [{"forecast_bin": str(k), **{c: round(float(v), 1) for c, v in r.items()}} for k, r in t.iterrows()]}
    return out


def weather_vs_model(g: pd.DataFrame) -> dict:
    """Same days, both weather variants (2024+): how much of the miss the weather forecast makes."""
    p = g[g["season"] >= 2024].pivot_table(index=["D", "lead"], columns="weather", values="F").dropna()
    p = p.join(g[(g["season"] >= 2024) & (g["weather"] == "actual")].set_index(["D", "lead"])[["A", "dA"]])
    out = {}
    for L in LEADS:
        x = p.xs(L, level="lead")
        for name, m in (("all", x["A"] > 0), ("rise30", x["dA"] >= RISE), ("rise50", x["dA"] >= 50), ("red", x["A"] > RED),
                        ("red_missed", (x["A"] > RED) & (x["forecast"] <= RED))):
            s = x[m]
            fb, ab = (s["forecast"] - s["A"]).mean(), (s["actual"] - s["A"]).mean()
            out[f"L{L}_{name}"] = {"n": int(len(s)), "actual": round(float(s["A"].mean())), "bias_forecast_weather": round(float(fb), 1),
                                   "bias_actual_weather": round(float(ab), 1),
                                   "weather_share": round(float((fb - ab) / fb), 2) if name != "all" else None}
    return out


def red_chance(g: pd.DataFrame) -> dict:
    """The site's red-day chance for days 1–7 (aqf.forecast: the daily log-error quantiles)."""
    B = json.loads((PROC / "intervals.json").read_text())["daily_q"]
    out = {}
    for (w, L), x in g.groupby(["weather", "lead"]):
        dprod, p = x["lead"].values + 1, np.full(len(x), np.nan)  # the site's day 1 is the issue day
        for b in B:
            m = (dprod >= b["lo"]) & (dprod <= b["hi"])
            p[m] = (np.array(b["q"])[None, :] > (np.log(150) - np.log(x["F"].values[m]))[:, None]).mean(axis=1)
        red = (x["A"].values > RED).astype(float)
        bs, bs0 = ((p - red) ** 2).mean(), ((red.mean() - red) ** 2).mean()
        t = pd.DataFrame({"p": p, "red": red}).groupby(pd.cut(p, [-0.01, 0.1, 0.3, 0.5, 0.7, 1.0]), observed=True).agg(
            chance=("p", "mean"), observed=("red", "mean"), n=("red", "size"))
        out[f"{w}_L{L}"] = {"brier_skill": round(float(1 - bs / bs0), 2),
                            "red_days_with_30pct": round(float((p[red == 1] >= 0.3).mean()), 2),
                            "red_share_when_30pct": round(float(red[p >= 0.3].mean()), 2),
                            "reliability": [{"bin": str(k), **{c: round(float(v), 2) for c, v in r.items()}} for k, r in t.iterrows()]}
    return out


def missed_vs_caught(g: pd.DataFrame, d: pd.DataFrame) -> dict:
    cols = ["A", "F", "Aknown", "ws", "calm", "dws", "t", "dt", "p", "dp", "rain", "dry", "inv", "stag_run"]
    out = {}
    for (w, L), x in g.groupby(["weather", "lead"]):
        x = x.join(d.drop(columns=["A", "dA"]), on="D")
        red = x["A"] > RED
        missed, caught = red & (x["F"] <= RED), red & (x["F"] > RED)
        out[f"{w}_L{L}"] = {k: {"n": int(m.sum()), "median": {c: round(float(x.loc[m, c].median()), 1) for c in cols},
                                "months": {int(a): int(b) for a, b in x.loc[m, "D"].dt.month.value_counts().sort_index().items()}}
                            for k, m in (("missed", missed), ("caught", caught))}
    return out


def residual_links(g: pd.DataFrame, d: pd.DataFrame) -> dict:
    """What the model misses even with the actual weather (1 day ahead), against what we measure."""
    d = d.copy()
    for c in ("no2", "rh", "inv", "inv925"):
        d[c + "_z"] = _z(d, [c])[c]
    x = g[(g["weather"] == "actual") & (g["lead"] == 1)].sort_values("D").join(d.drop(columns=["A", "dA"]), on="D")
    x["lr"] = np.log(x["A"] / x["F"])  # > 0: forecast too low
    x["yesterday_miss"] = x["A1"] - x.groupby("season")["F"].shift(1)
    rho = {}
    for c in ["ws", "calm", "inv_z", "inv925", "rh_z", "snow3", "no2_z", "t", "dt", "dp", "dow", "stag_run", "dry", "yesterday_miss"]:
        ok = x[[c, "lr"]].dropna()
        rho[c] = {"rho": round(float(spearmanr(ok[c], ok["lr"])[0]), 2), "n": int(len(ok))}
    x["r"] = x["A"] - x["F"]
    dry = x.groupby(pd.cut(x["dry"], [-1, 1, 3, 7, 14, 40]), observed=True)["r"].agg(["mean", "size"])
    return {"spearman": rho, "under_forecast_by_dry_days": {str(k): {"mean": round(float(r["mean"]), 1), "n": int(r["size"])} for k, r in dry.iterrows()}}


def causes(d: pd.DataFrame) -> dict:
    c = d[d.index.month.isin(COLD) & d["dA"].notna()]
    feats = ["ws", "calm", "gust", "dws", "t", "dt", "p", "dp", "rain", "dry", "rh", "cloud", "ws850", "inv", "stag_run"]
    z, rise, red = _z(c, feats), c["dA"] >= RISE, c["A"] > RED
    deep, cold = (c["calm"] >= 16) | (c["inv"] >= 4), c["t"] < 3
    return {"days": int(len(c)), "rise_days": int(rise.sum()), "red_days": int(red.sum()),
            "red_by_month": {int(a): int(b) for a, b in c[red].index.month.value_counts().sort_index().items()},
            "rise_profile": {f: {"rise_median": round(float(c.loc[rise, f].median()), 1), "other_median": round(float(c.loc[~rise, f].median()), 1),
                                 "rise_z": round(float(z.loc[rise, f].mean()), 2)} for f in feats},
            "red_share_deep_stagnation": round(float(deep[red].mean()), 2), "all_share_deep_stagnation": round(float(deep.mean()), 2),
            "red_share_cold": round(float(cold[red].mean()), 2), "all_share_cold": round(float(cold.mean()), 2),
            "p_red_deep_and_cold": round(float(red[deep & cold].mean()), 2), "p_red_deep": round(float(red[deep].mean()), 2),
            "p_red_neither": round(float(red[~deep & ~cold].mean()), 2)}


def hourly_peaks(A: pd.Series) -> dict:
    """Tomorrow's highest hour: forecast (06:00, archived weather) against measured."""
    hb = pd.read_parquet(PROC / "bt_hourly.parquet", columns=["origin", "pred_aqi", "actual_aqi", "T", "weather", "season"])
    o = hb[(hb["origin"].dt.hour == 6) & (hb["weather"] == "forecast") & (hb["season"] >= 2024)].copy()
    o["D"] = o["T"].dt.normalize()
    o = o[(o["D"] - o["origin"].dt.normalize()).dt.days == 1]
    m = o.groupby("D").agg(fmax=("pred_aqi", "max"), amax=("actual_aqi", "max"), n=("pred_aqi", "size"))
    m = m[(m["n"] == 24) & m.index.month.isin(COLD)]
    hi = m["amax"] > 200
    return {"days": int(len(m)), "forecast_max": round(float(m["fmax"].mean())), "actual_max": round(float(m["amax"].mean())),
            "days_over_200": int(hi.sum()), "forecast_max_on_them": round(float(m.loc[hi, "fmax"].mean())), "actual_max_on_them": round(float(m.loc[hi, "amax"].mean()))}


def main() -> dict:
    A = daily_aqi(load_target()[0]["pm"])
    g, d = lead_daily(A), daily_weather(A)
    out = {"errors_by_change": errors_by_change(g), "calibration": calibration(g), "weather_vs_model": weather_vs_model(g),
           "red_chance": red_chance(g), "missed_vs_caught": missed_vs_caught(g, d), "residual_links": residual_links(g, d),
           "causes": causes(d), "hourly_peaks": hourly_peaks(A)}
    (PROC / "peaks.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in out.items():
        print(f"\n== {k}\n" + json.dumps(v, ensure_ascii=False)[:3000])
    return out


if __name__ == "__main__":
    main()
