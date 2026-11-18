"""WeatherNext 3, step 1 — wind filter (rules: docs/checks/2026-11-07-weathernext.md, fixed before the data).

Is the WeatherNext 3 surface wind closer to the wind measured at Tashkent airport (METAR UZTT) than our
4-model mean? Same metrics and days as research/wind_model_skill.py: daily mean 10 m wind (r_wind) and
daily number of calm hours < 5 km/h (r_calm), October-March, lead 1 and 3 days.

WeatherNext 3 from Earth Engine (projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg,
0.1° cell containing 41.31 N 69.28 E): the run available at our 06:00 Tashkent issue on day D is the
12 UTC run of D-1 (published 20:10 UTC); target day D+L (local 19 UTC of the day before to 18 UTC) is
forecast hours 31-54 (L = 1) and 79-102 (L = 3). Ensemble mean wind_speed_10m_mean decides;
wind_speed_10m_p50 is reported only. Each run's point series is written once to
data/raw/weather/weathernext3/<YYYYMMDD>12.parquet.

Decision on winter 2024/25 only: pass if r_wind is higher than our mean's at both leads and r_calm is not
lower by more than 0.02. Winter 2025/26 and October 2026 are reported, not used.

Needs earthengine-api and the owner's Earth Engine login (`.venv/bin/earthengine authenticate`) with
an allowlisted account and a Cloud project registered for Earth Engine:

    WN_EE_PROJECT=<project-id> PYTHONPATH=. .venv/bin/python research/weathernext_wind.py fetch 2024
    WN_EE_PROJECT=<project-id> PYTHONPATH=. .venv/bin/python research/weathernext_wind.py fetch 2025   # reference
    PYTHONPATH=. .venv/bin/python research/weathernext_wind.py score                                    # -> data/processed/weathernext_wind.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from aqf.data import PROC, RAW, load_prev_runs_ensemble

COLLECTION = "projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg"
LAT, LON = 41.31, 69.28
BANDS = ["wind_speed_10m_mean", "wind_speed_10m_p50"]
HOURS = {1: (31, 54), 3: (79, 102)}
DIR = RAW / "weather" / "weathernext3"
METAR = RAW / "weather" / "metar_UZTT_full.csv"
COLD = [10, 11, 12, 1, 2, 3]
SEASONS = {2024: ("2024-10-01", "2025-03-31"), 2025: ("2025-10-01", "2026-03-31"), 2026: ("2026-10-01", "2026-11-12")}


def runs(season: int) -> list[pd.Timestamp]:
    """12 UTC inits whose lead-1 or lead-3 target days fall in the season."""
    a, b = (pd.Timestamp(x) for x in SEASONS[season])
    return list(pd.date_range(a - pd.Timedelta(days=4), b - pd.Timedelta(days=2), freq="D") + pd.Timedelta(hours=12))


def fetch_run(ee, col, init: pd.Timestamp) -> pd.DataFrame | None:
    path = DIR / f"{init:%Y%m%d%H}.parquet"
    if path.exists():  # written once
        return pd.read_parquet(path)
    sel = (col.filter(ee.Filter.eq("start_time", f"{init:%Y-%m-%dT%H:%M:%SZ}"))
              .filter(ee.Filter.rangeContains("forecast_hour", HOURS[1][0], HOURS[3][1]))
              .select(BANDS))
    rows = sel.getRegion(ee.Geometry.Point([LON, LAT]), 10000).getInfo()
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["valid_utc"] = pd.to_datetime(df["time"], unit="ms")
    df = df.drop(columns=["id", "time"]).sort_values("valid_utc").reset_index(drop=True)
    df["init_utc"] = init
    df["forecast_hour"] = ((df["valid_utc"] - init) / pd.Timedelta(hours=1)).round().astype(int)
    if not (df["forecast_hour"].between(HOURS[1][0], HOURS[3][1]).all() and df[BANDS].notna().any().all()):
        raise ValueError(f"unexpected content for init {init}")
    df.to_parquet(path)
    return df


def fetch(season: int) -> None:
    import ee
    ee.Initialize(project=os.environ["WN_EE_PROJECT"])
    col = ee.ImageCollection(COLLECTION)
    DIR.mkdir(parents=True, exist_ok=True)
    todo, t0, missing = runs(season), time.time(), []
    for i, init in enumerate(todo, 1):
        for attempt in range(3):
            try:
                if fetch_run(ee, col, init) is None:
                    missing.append(f"{init:%Y-%m-%d}")
                break
            except Exception as e:  # noqa: BLE001 - quota or network: wait and retry
                if attempt == 2:
                    print(f"{init:%Y-%m-%d %H}Z failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        if i % 20 == 0:
            print(f"{i}/{len(todo)} runs, {time.time() - t0:.0f}s", flush=True)
    print(f"season {season}: {len(todo)} runs, missing {len(missing)} {missing[:5]}", flush=True)


def wn3_hourly(lead: int, band: str) -> pd.Series:
    """Hourly local-time series of the given lead day from the saved runs."""
    parts = []
    lo, hi = HOURS[lead]
    for p in sorted(DIR.glob("??????????.parquet")):
        d = pd.read_parquet(p)
        d = d[d["forecast_hour"].between(lo, hi)]
        parts.append(pd.Series(d[band].values * 3.6, index=d["valid_utc"] + pd.Timedelta(hours=5)))  # m/s -> km/h, UTC+5
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()]


def score() -> dict:
    m = pd.read_csv(METAR, na_values="M", usecols=["valid", "sknt"])
    m.index = pd.to_datetime(m.pop("valid"))
    ap = (m["sknt"] * 1.852).resample("h").mean()
    calm = lambda s: (s < 5).astype(float).where(s.notna()).resample("D").sum(min_count=20)  # noqa: E731
    ens = load_prev_runs_ensemble()
    out = {}
    for season, (a, b) in SEASONS.items():
        res = {}
        for lead in (1, 3):
            series = {"wn3_mean": wn3_hourly(lead, BANDS[0]), "wn3_p50": wn3_hourly(lead, BANDS[1]),
                      "ours": ens[lead]["wind_speed_10m"].astype(float)}
            d = pd.DataFrame({k: v.resample("D").mean() for k, v in series.items()} | {"ap": ap.resample("D").mean()}
                             | {f"{k}_calm": calm(v) for k, v in series.items()} | {"ap_calm": calm(ap)})
            d = d[(d.index >= a) & (d.index <= b) & d.index.month.isin(COLD)].dropna()
            res[lead] = {"days": int(len(d))} | {
                k: {"r_wind": round(float(d[k].corr(d["ap"])), 3), "r_calm": round(float(d[f"{k}_calm"].corr(d["ap_calm"])), 3),
                    "bias_kmh": round(float((d[k] - d["ap"]).mean()), 2)} for k in series}
        out[season] = res
    s24 = out[2024]
    out["passes_step1"] = bool(all(s24[L]["wn3_mean"]["r_wind"] > s24[L]["ours"]["r_wind"]
                                   and s24[L]["wn3_mean"]["r_calm"] >= s24[L]["ours"]["r_calm"] - 0.02 for L in (1, 3)))
    (PROC / "weathernext_wind.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        fetch(int(sys.argv[2]))
    else:
        print(json.dumps(score(), indent=1))
