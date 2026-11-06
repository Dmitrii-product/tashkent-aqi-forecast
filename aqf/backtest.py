"""Walk-forward backtest over cold seasons (October–March).

For each test season the model is trained only on data that ends before 1 October of that
season, then issues forecasts from origins inside the season and is scored against what
actually happened. Two weather variants:
  actual   — reanalysis weather (upper bound: perfect weather forecast)
  forecast — archived Open-Meteo forecasts as issued 1–7 days ahead (realistic; 2024+)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import aqi
from .data import (PROC, RAW, load_prev_runs, load_prev_runs_ensemble, load_target, load_weather,
                   splice_forecast_weather)
from .features import daily_weather, weather_features
from .models import DailyModel, HourlyModel, daily_aqi, make_hourly_model

H_BUCKETS = [(1, 3), (4, 6), (7, 12), (13, 24), (25, 48), (49, 72), (73, 120), (121, 168)]
D_BUCKETS = [(1, 1), (2, 3), (4, 7), (8, 14), (15, 30), (31, 60), (61, 90)]


def season_bounds(y: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{y}-10-01"), pd.Timestamp(f"{y + 1}-04-01")


def score(df: pd.DataFrame, key: str, buckets, pred_cols) -> pd.DataFrame:
    rows = []
    for lo, hi in buckets:
        sub = df[(df[key] >= lo) & (df[key] <= hi)]
        if sub.empty:
            continue
        r = {"bucket": f"{lo}-{hi}", "n": len(sub)}
        a = sub["actual_aqi"].values
        for c in pred_cols:
            p = sub[c].values
            ok = ~np.isnan(p) & ~np.isnan(a)
            e = p[ok] - a[ok]
            r[f"{c}_mae"] = np.abs(e).mean()
            r[f"{c}_bias"] = e.mean()
            r[f"{c}_cat"] = (aqi.category_index(p[ok]) == aqi.category_index(a[ok])).mean()
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- hourly
def backtest_hourly(seasons: list[int], max_h: int = 168, step_hours: int = 3,
                    retrain: str = "MS") -> pd.DataFrame:
    """Walk forward through each season, retraining at every `retrain` boundary (month
    starts by default) on all data before it, the current winter's first months included,
    as production does (it retrains weekly)."""
    wx = load_weather()
    wf = weather_features(wx)
    prev = load_prev_runs_ensemble()
    wf_fc = splice_forecast_weather(wx, prev)
    prev_start = min(f.index.min() for f in prev.values())
    choice = json.loads((PROC / "horizon_choice.json").read_text())
    cams = pd.read_parquet(RAW / "weather" / "cams.parquet")
    res = []
    for y in seasons:
        t0 = time.time()
        start, end = season_bounds(y)
        n_orig = 0
        cutoffs = pd.date_range(start, end, freq=retrain, inclusive="left")
        for i, cut in enumerate(cutoffs):
            nxt = cutoffs[i + 1] if i + 1 < len(cutoffs) else end
            tgt, cal = load_target(profile_cutoff=cut)
            pm, src = tgt["pm"], tgt["src"]
            last = min(nxt, pm.dropna().index.max())
            origins = pd.date_range(cut, last, freq=f"{step_hours}h", inclusive="left")
            origins = origins[np.log1p(pm).ffill(limit=2).reindex(origins).notna().values]
            if not len(origins):
                continue
            m = make_hourly_model(choice, cams).fit(pm, src, wf_fc, cutoff=cut)
            variants = {"actual": wf}
            if cut >= prev_start + pd.Timedelta(days=8):
                variants["forecast"] = wf_fc
            for name, w in variants.items():
                p = m.predict_frame(origins, pm, w).reset_index()
                p["T"] = p["origin"] + pd.to_timedelta(p["h"], unit="h")
                p["actual_aqi"] = aqi.pm_to_aqi(pm.reindex(p["T"]).values)
                p["src"] = src.reindex(p["T"]).values
                p["season"] = y
                p["weather"] = name
                p["trained_until"] = cut
                res.append(p.dropna(subset=["actual_aqi"]))
            n_orig += len(origins)
        print(f"  season {y}/{y + 1}: {n_orig} origins, {len(cutoffs)} retrains, "
              f"{time.time() - t0:.0f}s", flush=True)
    return pd.concat(res, ignore_index=True)


# ------------------------------------------------------------------------- daily
def backtest_daily(seasons: list[int], wx_days: int = 7, rounds: int = 500) -> pd.DataFrame:
    wx = load_weather()
    dw = daily_weather(wx)
    res = []
    for y in seasons:
        start, end = season_bounds(y)
        pm = load_target(profile_cutoff=start)[0]["pm"]
        a = daily_aqi(pm)
        m = DailyModel(wx_days=wx_days, rounds=rounds).fit(pm, wx, cutoff=start)
        # origin = last complete day; first origin is 30 Sep -> first target 1 Oct
        origins = pd.date_range(start - pd.Timedelta(days=1), end - pd.Timedelta(days=2), freq="D")
        origins = origins[origins < a.dropna().index.max()]
        p = m.predict_frame(origins, pm, dw).reset_index()
        p["T"] = p["origin"] + pd.to_timedelta(p["d"], unit="D")
        p["actual_aqi"] = a.reindex(p["T"]).values
        p["season"] = y
        res.append(p.dropna(subset=["actual_aqi"]))
        print(f"  daily season {y}/{y + 1}: {len(origins)} origins", flush=True)
    return pd.concat(res, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2020,2021,2022,2023,2024,2025,2026")
    ap.add_argument("--what", default="hourly,daily")
    a = ap.parse_args()
    seasons = [int(s) for s in a.seasons.split(",")]
    PROC.mkdir(parents=True, exist_ok=True)
    if "hourly" in a.what:
        print("hourly model (1..168 h)")
        new = backtest_hourly(seasons)
        out = PROC / "bt_hourly.parquet"
        if out.exists():  # replace only the seasons that were re-run
            old = pd.read_parquet(out)
            new = pd.concat([old[~old["season"].isin(seasons)], new], ignore_index=True)
        new.to_parquet(out)
    if "daily" in a.what:
        print("daily 90-day model")
        backtest_daily(seasons).to_parquet(PROC / "bt_daily.parquet")
