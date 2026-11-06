"""Why did the forecast miss 28.10–3.11 2026 so badly? (owner, 2026-11-04)

The committee week (docs/checks/2026-10-27-committee-vs-forecast.md): the air stayed at 150–186 while
the forecast kept expecting a front; the error breakdown (aqf.attribution) put most of it on the
weather forecast (-15 AQI at 1 day, -33 at 3 days, every day the same sign). Here: what exactly in
the weather forecast differed from what happened, at which hours the forecast was low, and whether
anything the model does not see (NO2 of local combustion, fires, dust) was unusual for these dates.

  1. weather as issued (the saved <stem>_wx.parquet of the morning forecasts, mean of 4 models)
     against the archive (ERA5) for each day at 1 and 3 days ahead, next to the usual error of the
     same mean at that lead in Oct–Mar 2024–2026 (archived forecasts);
  2. the 1-day forecast's error by hour of day in that week, against the backtest's for Oct–Nov;
  3. this week against the same dates of earlier years: AQI, NO2 (Sentinel-5P), fires (FIRMS,
     200 km), CAMS dust;
  4. the wind measured at Tashkent airport (METAR UZTT via the Iowa Environmental Mesonet, every
     30 min) against ERA5 and the forecasts; whether the wind forecast's error lasts from day to day.

    PYTHONPATH=. .venv/bin/python research/late_october_miss.py

Result (2026-11-04):
  1. The weather forecast was too windy on every one of the 14 (day, lead) pairs: 10 m wind +1.3 km/h
     at 1 day and +1.6 at 3 days (usual error 0.0 ± 1.4), 6 and 10 calm hours a day fewer than
     happened (usual -0.8 ± 5). Temperature, humidity and rain as usual; night inversions slightly
     stronger than forecast (-0.8 °C at 1 day). Actual ERA5 wind 3.9 km/h against 5.7 usual.
  2. The 1-day forecast was low in the daytime: -13…-31 AQI from 09 to 17 h (usual -2…-7), near zero
     at night — the model expected the usual daytime clearing, which weak wind did not bring.
  3. AQI 160 for 28.10–3.11, the highest of 2019–2026 for these dates (80–141); 1–27.10 was normal
     (105). NO2 336, +44% on early October, but 2024 had 363 with AQI 122. Fires within 200 km 16.5 a
     day, the most for these dates, yet 19 a day in 1–27.10 with normal air. CAMS dust 6.6, middling.
     Central heating was not on yet (not announced as of 26.10; 3–7.11 in 2023–2025).
  4. Airport wind 7.4 km/h against 9.0 usual (ERA5 3.9 / 5.7): the calm is real, not an ERA5
     artefact. The airport wind tracks the daily AQI a little better than ERA5 (Spearman -0.62 / -0.55;
     visibility -0.62). The wind forecast's daily error lasts: autocorrelation 0.40 (1 day ahead) and
     0.39 (3 days); the error already known at issue time correlates 0.22–0.29 with the coming one.
  Cause: a stagnant spell that the weather models kept breaking up too early, plus a dry October.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import PROC, RAW, load_prev_runs_ensemble, load_target, load_weather
from aqf.live import issued_stems
from aqf.models import daily_aqi
from aqf.satellite import fire_series, no2_series

LOG = PROC.parent / "forecast_log"
DAYS = pd.date_range("2026-10-28", "2026-11-03")
COLD = [10, 11, 12, 1, 2, 3]


def daily(w: pd.DataFrame) -> pd.DataFrame:
    w = w.astype(float)
    day = lambda s, how="mean": getattr(s.resample("D"), how)()  # noqa: E731
    d = pd.DataFrame({"ws": day(w["wind_speed_10m"]), "calm": day((w["wind_speed_10m"] < 5).astype(float), "sum"),
                      "t": day(w["temperature_2m"]), "rh": day(w["relative_humidity_2m"]), "cloud": day(w["cloud_cover"]),
                      "sw": day(w["shortwave_radiation"]), "rain": day(w["precipitation"], "sum"), "p": day(w["surface_pressure"])})
    if "temperature_850hPa" in w:
        inv = w["temperature_850hPa"] - w["temperature_2m"]
        d["inv_night"] = day(inv.where(w.index.hour < 9))
        d["inv_day"] = day(inv.where((w.index.hour >= 11) & (w.index.hour <= 17)))
    return d


def weather_miss() -> pd.DataFrame:
    act = daily(load_weather())
    stems = {pd.Timestamp(s[:8]): s for s in issued_stems()}
    rows = []
    for L in (1, 3):
        for D in DAYS:
            s = stems.get(D - pd.Timedelta(days=L))
            if s is None or not (LOG / f"{s}_wx.parquet").exists():
                continue
            fc = daily(pd.read_parquet(LOG / f"{s}_wx.parquet"))
            if D in fc.index and D in act.index:
                rows.append({"lead": L, "day": D, **{f"{c}": round(float(fc.loc[D, c] - act.loc[D, c]), 2) for c in fc.columns}})
    week = pd.DataFrame(rows)
    # the usual error of the same 4-model mean at that lead, Oct–Mar 2024–2026
    usual = {}
    for L in (1, 3):
        f = daily(load_prev_runs_ensemble()[L])
        e = (f - act.reindex(f.index)).dropna(how="all")
        e = e[e.index.month.isin(COLD) & (e.index >= "2024-10-01") & (e.index < "2026-10-01")]
        usual[L] = e
    print("1. weather as issued minus actual, daily means (lead, day); then the week's mean against the usual error")
    print(week.round(1).to_string(index=False))
    for L in (1, 3):
        w = week[week["lead"] == L].drop(columns=["lead", "day"]).mean()
        u = usual[L]
        print(f"\n lead {L}: this week   " + "  ".join(f"{c} {w[c]:+.1f}" for c in w.index))
        print(f" lead {L}: usual mean  " + "  ".join(f"{c} {u[c].mean():+.1f}" if c in u else f"{c}   n/a" for c in w.index))
        print(f" lead {L}: usual sd    " + "  ".join(f"{c} {u[c].std():.1f}" if c in u else f"{c}  n/a" for c in w.index))
    return week


def hourly_miss() -> None:
    pm = load_target()[0]["pm"]
    obs = pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)
    rows = []
    for s in issued_stems():
        js = json.loads((LOG / f"{s}.json").read_text())
        issue = pd.Timestamp(js["origin"]).normalize()
        if issue + pd.Timedelta(days=1) not in DAYS:
            continue
        for r in js.get("hourly_7d_gbm", js["hourly_7d"]):
            t = pd.Timestamp(r["t"])
            if t.normalize() == issue + pd.Timedelta(days=1):
                rows.append({"hour": t.hour, "err": r["aqi"] - obs.get(t, np.nan)})
    wk = pd.DataFrame(rows).dropna().groupby("hour")["err"].mean()
    bt = pd.read_parquet(PROC / "bt_hourly.parquet", columns=["origin", "T", "pred_aqi", "actual_aqi", "weather", "season"])
    bt = bt[(bt["weather"] == "forecast") & (bt["origin"].dt.hour == 6) & bt["T"].dt.month.isin([10, 11])
            & ((bt["T"].dt.normalize() - bt["origin"].dt.normalize()).dt.days == 1)]
    usual = (bt["pred_aqi"] - bt["actual_aqi"]).groupby(bt["T"].dt.hour).mean()
    print("\n2. the 1-day forecast's error by hour of day (forecast - actual): this week / backtest Oct–Nov")
    print(pd.DataFrame({"week": wk.round(0), "usual": usual.round(0)}).T.to_string())


def same_dates() -> None:
    A = daily_aqi(load_target()[0]["pm"])
    no2 = no2_series()["no2"] if len(no2_series()) else pd.Series(dtype=float)
    fires = fire_series()["fires"]
    cams = pd.read_parquet(RAW / "weather" / "cams.parquet")["dust"].resample("D").mean()
    print("\n3. 28.10–3.11 of each year (and 1–27.10 before it): daily AQI, NO2 (µmol/m²), fires within 200 km, CAMS dust (µg/m³)")
    rows = []
    for y in range(2018, 2027):
        wk = pd.date_range(f"{y}-10-28", f"{y}-11-03")
        before = pd.date_range(f"{y}-10-01", f"{y}-10-27")
        sel = lambda s, idx: float(s.reindex(idx).mean()) if s.reindex(idx).notna().sum() >= 3 else np.nan  # noqa: E731
        rows.append({"year": y, "aqi": sel(A, wk), "aqi_oct1_27": sel(A, before), "no2": sel(no2, wk), "no2_oct1_27": sel(no2, before),
                     "fires": sel(fires, wk), "fires_oct1_27": sel(fires, before), "dust": sel(cams, wk)})
    print(pd.DataFrame(rows).round(1).to_string(index=False))


METAR = RAW / "weather" / "metar_UZTT.csv"
IEM = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=UZTT&data=sknt&data=drct&data=vsby"
       "&data=tmpc&data=relh&data=wxcodes&year1=2024&month1=10&day1=1&year2=2026&month2=11&day2=4"
       "&tz=Asia%2FTashkent&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4")


def airport() -> None:
    import requests
    if not METAR.exists():
        r = requests.get(IEM, timeout=300)
        r.raise_for_status()
        METAR.write_text(r.text)
    m = pd.read_csv(METAR, na_values="M")
    m = m.set_index(pd.to_datetime(m["valid"]))
    h = (m["sknt"] * 1.852).resample("h").mean()
    era, f1 = load_weather(), load_prev_runs_ensemble()[1]
    day = lambda s, how="mean": getattr(s.resample("D"), how)()  # noqa: E731
    D = pd.DataFrame({"airport": day(h), "era5": day(era["wind_speed_10m"]), "forecast_1d": day(f1["wind_speed_10m"]),
                      "aqi": daily_aqi(load_target()[0]["pm"])})
    cold = D[D.index.month.isin(COLD) & (D.index >= "2024-10-01") & (D.index < "2026-10-28")]
    print("\n4. 10 m wind, km/h: this week / Oct–Mar 2024–26")
    print(pd.DataFrame({"week": D.loc[DAYS].mean(), "usual": cold.mean()}).round(1).T.to_string())


if __name__ == "__main__":
    weather_miss()
    hourly_miss()
    same_dates()
    airport()
