"""Would the boundary-layer (mixing) height make the hourly forecast better?

The peaks study (research/peaks_study.py, 2026-10-30) found that the red days the forecast misses
start in moderate weather with a weak 850 hPa inversion; a shallow layer of stagnant air near the
ground may not show at 850 hPa. The owner chose to test the mixing height as an input (2026-10-30).

What exists (checked 2026-10-30, Open-Meteo, Tashkent):
  * ERA5 reanalysis (archive API): boundary_layer_height hourly over the whole record, ~5 days late;
  * near-analysis archive (historical-forecast API): only since 2024-09-01;
  * archived forecasts by lead time (previous-runs API): none;
  * live forecast: best_match and GFS (they differ by up to 3x in the evening).
So no honest "as forecast" history exists. This test gives the model ERA5's mixing height at every
lead: perfect knowledge of it, an upper bound of what it could add. All other inputs are as in
production (archived weather forecasts by lead time).

Rules fixed before the run (as for NO2 and westerly wind):
  * features at the target hour and looking back from it, from ERA5:
      blh       — log mixing height at the hour, 24 h mean of its log, highest in the last 24 h
      vent      — ventilation: log(mixing height x 10 m wind speed) at the hour and its 24 h mean
                  (wind from the weather the model gets)
      blh_vent  — both
  * head-to-head: trained once on everything before 1 October, forecasts every 3 h through the
    season, all 14 horizon groups with their production settings, 2 seeds averaged; the current
    model is model version 2026-10-28 (with westerly wind).
  * choice on winter 2024/25 only: the variant with the lowest MAE over all (origin, hour) rows;
    it passes only if it beats the current model there and then on both 2025/26 and October 2026
    (95% bootstrap interval over days of the MAE difference). October 2026 ends 7 days before
    ERA5's last hour, so every lead has the input.
  * as the gain is an upper bound: no gain -> rejected; a gain -> the next step is a realistic test
    (the forecast mixing height, which needs an archive of it first).

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \\
        PYTHONPATH=. .venv/bin/python research/blh_input_test.py      # ~1.5 h

Writes data/raw/weather/blh_era5.parquet (ERA5 mixing height), data/processed/blh_test.parquet
(every forecast row) and blh_test.json (summary).

Result (2026-10-30), MAE change vs the current model (95% interval over days), ERA5 mixing height:
                     1–24 h                 25–72 h                73–168 h               1–168 h
  2024/25 blh_vent   +0.05 (-0.21..+0.30)   -0.16 (-0.58..+0.24)   -0.80 (-1.11..-0.50)   -0.49 (-0.72..-0.28)  <- chosen
  2025/26 blh_vent   +0.14 (-0.05..+0.36)   -0.24 (-0.54..+0.08)   -0.50 (-0.80..-0.20)   -0.33 (-0.54..-0.13)
  Oct 2026 blh_vent  -0.32 (-0.68..+0.04)   -0.44 (-1.31..+0.48)   +0.01 (-1.20..+1.19)   -0.18 (-0.90..+0.55)
(blh alone: -0.45 / -0.19 / -0.18; vent alone: -0.36 / -0.12 / +0.02.) Under-forecast on hours
with AQI > 150: -34.4 -> -32.2 (2024/25), -38.4 -> -37.2 (2025/26), -21.1 -> -22.9 (Oct 2026).
Parked, not taken further: the whole gain is at 3–7 days, where this test hands the model the
true future mixing height (perfect weather for everything is worth -3.0 / -1.5 there in the
backtest); at 1–24 h, where a forecast of it is close to the truth, nothing. A forecast mixing
height is computed from the same forecast fields the model already has, so a realistic version
would add about nothing, and it does not touch the peaks (red hours stay ~35 too low).
Daily misses vs ERA5 mixing height (research/peaks_study.py rows, actual weather): Spearman -0.09
at 1 day, -0.16 at 3 days; missed red days mix deeper by day than caught ones (545 vs 363 m).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import PROC, RAW, load_prev_runs_ensemble, load_weather
from aqf.features import weather_features
from aqf.models import H_GROUPS, HOURLY_PARAMS, HOURLY_VARIANTS, _gbm
from aqf.sources import openmeteo
from research.no2_input_test import boot
from research.westerly_wind_test import split

BLH = RAW / "weather" / "blh_era5.parquet"
CHOICE = json.loads((PROC / "horizon_choice.json").read_text())
SEEDS = (0, 1)
VARIANTS = {
    "blh": ["x_lblh", "x_lblh24", "x_blhmax24"],
    "vent": ["x_lvent", "x_lvent24"],
    "blh_vent": ["x_lblh", "x_lblh24", "x_blhmax24", "x_lvent", "x_lvent24"],
}
EXTRA = sorted({c for v in VARIANTS.values() for c in v})


def fetch_blh() -> pd.Series:
    if BLH.exists():
        return pd.read_parquet(BLH)["boundary_layer_height"]
    frames = []
    for y in range(2018, pd.Timestamp.now().year + 1):
        js = openmeteo._get("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": openmeteo.LAT, "longitude": openmeteo.LON, "hourly": "boundary_layer_height",
            "start_date": f"{y}-01-01", "end_date": min(f"{y}-12-31", f"{pd.Timestamp.now():%Y-%m-%d}"),
            "timezone": openmeteo.TZ, "models": "era5"})  # ERA5 only: recent days are not filled from IFS
        frames.append(openmeteo._to_frame(js))
        time.sleep(2)
    df = pd.concat(frames).dropna()
    df.to_parquet(BLH)
    return df["boundary_layer_height"]


def extras(w: pd.DataFrame, blh: pd.Series) -> pd.DataFrame:
    """The candidate inputs, per hour, looking back from it."""
    b = blh.reindex(w.index).interpolate(limit=3).clip(lower=10)
    lb = np.log(b)
    lv = np.log(b * w["wind_speed_10m"].clip(lower=0.5) / 3.6)  # m2/s
    f = pd.DataFrame(index=w.index)
    f["x_lblh"] = lb
    f["x_lblh24"] = lb.rolling(24, min_periods=12).mean()
    f["x_blhmax24"] = b.rolling(24, min_periods=12).max()
    f["x_lvent"] = lv
    f["x_lvent24"] = lv.rolling(24, min_periods=12).mean()
    return f


def weather_by_lead(archive: pd.DataFrame, prev: dict[int, pd.DataFrame], blh: pd.Series) -> dict[int, pd.DataFrame]:
    """aqf.data.splice_forecast_weather plus the candidate columns."""
    out = {}
    for d, f in prev.items():
        w = archive.astype(float)
        common, cols = w.index.intersection(f.index), list(f.columns)
        w.loc[common, cols] = f.loc[common, cols].astype(float).fillna(w.loc[common, cols]).values
        out[d] = weather_features(w).join(extras(w, blh))
    return out


def seasons(blh: pd.Series) -> dict:
    last = blh.dropna().index.max() - pd.Timedelta(days=7)
    return {2024: ("2024-10-01", "2025-04-01"), 2025: ("2025-10-01", "2026-04-01"), 2026: ("2026-10-01", f"{last:%Y-%m-%d %H:00}")}


def run() -> pd.DataFrame:
    blh = fetch_blh()
    wf_fc = weather_by_lead(load_weather(), load_prev_runs_ensemble(), blh)
    cams_all = pd.read_parquet(RAW / "weather" / "cams.parquet")
    extra = [f"w_{c}" for c in EXTRA]
    res = []
    for y, (cut, end) in seasons(blh).items():
        for group in H_GROUPS:
            t0 = time.time()
            kw = HOURLY_VARIANTS[CHOICE[f"{group[0]}-{group[1]}"]]
            cams = cams_all if kw.get("use_cams") else None
            X, yy, w, Xv, meta = split(group, pd.Timestamp(cut), end, wf_fc, cams)
            base = [c for c in X.columns if X[c].notna().any() and c not in extra]
            out = meta.assign(season=y, group=f"{group[0]}-{group[1]}")
            todo = {"current": base} | {k: base + [f"w_{c}" for c in v] for k, v in VARIANTS.items()}
            for name, cols in todo.items():
                preds = [_gbm(HOURLY_PARAMS, kw.get("rounds", 300), seed, X[cols], yy, w, "lgbm", kw.get("lgbm_extra"))
                         .predict(Xv[cols].values) + Xv["lag0"].values for seed in SEEDS]
                out[name] = aqi.pm_to_aqi(np.clip(np.expm1(np.mean(preds, axis=0)), 0, None))
            res.append(out)
            maes = {k: np.abs(out[k] - out["actual"]).mean() for k in todo}
            print(f"{y} {group}: " + "  ".join(f"{k} {v:.2f}" for k, v in maes.items())
                  + f"  ({len(out)} rows, {time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(PROC / "blh_test.parquet")
    return df


def summarise(df: pd.DataFrame) -> dict:
    bands = {"1-24": (1, 24), "25-72": (25, 72), "73-168": (73, 168), "1-168": (1, 168)}
    mae = lambda d, k: float(np.abs(d[k] - d["actual"]).mean())  # noqa: E731
    ys = sorted(df["season"].unique())
    sel = df[df["season"] == 2024]
    chosen = min(VARIANTS, key=lambda k: mae(sel, k))
    red = df["actual"] > 150
    out = {"mae": {int(y): {k: {b: round(mae(df[(df["season"] == y) & df["h"].between(*r)], k), 2) for b, r in bands.items()}
                            for k in ("current", *VARIANTS)} for y in ys},
           "chosen_on_2024": chosen, "chosen_beats_current_2024": mae(sel, chosen) < mae(sel, "current"),
           "diff_vs_current": {int(y): {k: {b: [round(x, 2) for x in boot(df[(df["season"] == y) & df["h"].between(*r)], "current", k)]
                                            for b, r in bands.items()} for k in VARIANTS} for y in ys},
           # the hours the peaks study is about: is the model less low on them?
           "bias_on_red_hours": {int(y): {k: round(float((df.loc[(df["season"] == y) & red, k] - df.loc[(df["season"] == y) & red, "actual"]).mean()), 1)
                                          for k in ("current", *VARIANTS)} for y in ys},
           "rows": {int(y): int((df["season"] == y).sum()) for y in ys}}
    print(json.dumps(out, indent=1))
    (PROC / "blh_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    df = pd.read_parquet(PROC / "blh_test.parquet") if "--summary" in sys.argv else run()
    summarise(df)
