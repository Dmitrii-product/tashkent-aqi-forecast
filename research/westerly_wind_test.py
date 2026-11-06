"""Would "westerly wind" (and its neighbour, dry soil) make the hourly forecast better?

IQAir's note of 2026-10-26 says strong westerly winds bring dust from the Kyzylkum and Kazakhstan
and cause the sharp PM2.5 spikes; the owner asked whether that is a useful input (2026-10-28).
The model already sees the wind at each target hour (u, v at 10 m and 850 hPa, gusts, 24 h mean
speed); what it does not see is how long and how hard the wind has been blowing from the west
before that hour, or how dry the ground is (it remembers rain for 72 h only).

Rules fixed before the run (as for NO2, research/no2_input_test.py):
  * features from the same weather the model gets (archived forecasts by lead time, reanalysis
    before 2024; upper air from the near-analysis archive, as for the existing 850 hPa inputs),
    each looking back from the target hour:
      west      — 24 h mean westerly wind component at 10 m and 850 hPa, strongest gust from the
                  west (225–315°) in the last 24 h
      dry       — days since the last day with ≥ 1 mm of rain (capped at 40)
      west_dry  — both
  * head-to-head: trained once on everything before 1 October, forecasts every 3 h through the
    season, all 14 horizon groups with their production settings, 2 seeds averaged.
  * choice on winter 2024/25 only: the variant with the lowest MAE over all (origin, hour) rows;
    accepted only if it beats the current model there and then on both 2025/26 and October 2026
    (95% bootstrap interval over days of the MAE difference).

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
        PYTHONPATH=. .venv/bin/python research/westerly_wind_test.py      # ~1.5 h

Writes data/processed/west_test.parquet (every forecast row) and west_test.json (summary).

Result (2026-10-28), MAE change vs the current model over 1–168 h (95% interval over days):
                 2024/25 (choice)       2025/26                Oct 2026
  west           -0.49 (-0.67..-0.32)   -0.75 (-0.94..-0.56)   -0.19 (-0.50..+0.12)  <- chosen
  dry            -0.02 (-0.11..+0.07)   -0.24 (-0.32..-0.16)   -0.10 (-0.44..+0.24)
  west_dry       -0.48                  -0.78                  -0.84
  west_sfc       -0.42 (-0.60..-0.23)   -0.67 (-0.86..-0.47)   -0.56 (-0.94..-0.20)  (post hoc)
"west" gained at 1–72 h everywhere but lost at 73–168 h in October 2026 (+0.49, +0.09..+0.86);
without the 850 hPa wind ("west_sfc") nothing is lost anywhere. The surface version then went
through the full protocol: research/westerly_wind_full.py.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import PROC, RAW, load_prev_runs_ensemble, load_target, load_weather
from aqf.features import Climatology, hourly_rows, pm_state, same_hour_mean, target_times, weather_features
from aqf.models import H_GROUPS, HOURLY_PARAMS, HOURLY_VARIANTS, _gbm, _weights
from research.no2_input_test import boot

SEASONS = {2024: ("2024-10-01", "2025-04-01"), 2025: ("2025-10-01", "2026-04-01"), 2026: ("2026-10-01", None)}
CHOICE = json.loads((PROC / "horizon_choice.json").read_text())
SEEDS = (0, 1)
VARIANTS = {
    "west": ["x_u24", "x_u850_24", "x_wgust24"],
    "dry": ["x_dry"],
    "west_dry": ["x_u24", "x_u850_24", "x_wgust24", "x_dry"],
}
# added after the run, not a candidate for the choice: the chosen "west" without the 850 hPa wind,
# which history takes from the near-analysis archive (no archived upper-air forecasts exist), so it
# may know too much beyond the first day; the surface wind comes from real archived forecasts
POSTHOC = {"west_sfc": ["x_u24", "x_wgust24"]}
EXTRA = sorted({c for v in (*VARIANTS.values(), *POSTHOC.values()) for c in v})


def extras(w: pd.DataFrame) -> pd.DataFrame:
    """The candidate inputs, per hour, looking back from it."""
    w = w.asfreq("h").interpolate(limit=3)
    f = pd.DataFrame(index=w.index)
    u = -w["wind_speed_10m"] * np.sin(np.deg2rad(w["wind_direction_10m"]))  # > 0: from the west
    u850 = -w["wind_speed_850hPa"] * np.sin(np.deg2rad(w["wind_direction_850hPa"]))
    west = w["wind_direction_10m"].between(225, 315)
    f["x_u24"] = u.rolling(24, min_periods=12).mean()
    f["x_u850_24"] = u850.rolling(24, min_periods=12).mean()
    f["x_wgust24"] = w["wind_gusts_10m"].where(west, 0.0).rolling(24, min_periods=12).max()
    rain = w["precipitation"].resample("D").sum() >= 1.0
    run = (~rain).astype(int).groupby(rain.cumsum()).cumsum()  # dry days in a row, ending that day
    f["x_dry"] = run.shift(1).clip(upper=40).reindex(w.index.normalize()).to_numpy()  # up to the day before
    return f


def weather_by_lead(archive: pd.DataFrame, prev: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """aqf.data.splice_forecast_weather plus the candidate columns."""
    out = {}
    for d, f in prev.items():
        w = archive.astype(float)
        common, cols = w.index.intersection(f.index), list(f.columns)
        w.loc[common, cols] = f.loc[common, cols].astype(float).fillna(w.loc[common, cols]).values
        out[d] = weather_features(w).join(extras(w))
    return out


def split(group, cut, end, wf_fc, cams, val_step=3):
    """Training rows before `cut` (as HourlyModel.fit), validation forecasts every `val_step` h."""
    lo, hi = group
    tgt, _ = load_target(profile_cutoff=cut)
    pm, src = tgt["pm"], tgt["src"]
    lp_all = np.log1p(pm)
    lp = lp_all[lp_all.index < cut]
    clim = Climatology().fit(lp)
    state, sh7 = pm_state(lp), same_hour_mean(lp)
    rng = np.random.default_rng(0)
    origins = state.index[(state.index.hour % 2 == 0) & state["lag0"].notna()]
    k = min(4, hi - lo + 1)
    H = rng.integers(lo, hi + 1, size=(len(origins), k)) if hi > lo else np.full((len(origins), 1), lo)
    X = hourly_rows(origins, H, lp, state, sh7, clim, wf_fc, src, loo=True, cams=cams)
    T = target_times(X)
    y = lp.reindex(T).values - X["lag0"].values
    ok = ~np.isnan(y)
    X, y, T = X[ok].reset_index(drop=True), y[ok], T[ok]
    w = np.asarray(_weights(T, X["src"].values), float)
    end = min(pd.Timestamp(end) if end else pm.dropna().index.max(), pm.dropna().index.max() - pd.Timedelta(hours=1))
    vo = pd.date_range(cut, end, freq=f"{val_step}h")
    vo = vo[lp_all.ffill(limit=2).reindex(vo).notna().values]
    Xv = hourly_rows(vo, np.arange(lo, hi + 1), lp_all, pm_state(lp_all), same_hour_mean(lp_all), clim, wf_fc, None, cams=cams)
    Xv["src"] = 1.0
    act = aqi.pm_to_aqi(pm.reindex(target_times(Xv)).values)
    okv = ~np.isnan(act) & ~np.isnan(Xv["lag0"].values)
    meta = pd.DataFrame({"origin": Xv.index.get_level_values("origin")[okv], "h": Xv.index.get_level_values("h")[okv],
                         "actual": act[okv]})
    return X, y, w, Xv[okv].reset_index(drop=True), meta


def run(variants: dict = VARIANTS, into: pd.DataFrame | None = None) -> pd.DataFrame:
    wf_fc = weather_by_lead(load_weather(), load_prev_runs_ensemble())
    cams_all = pd.read_parquet(RAW / "weather" / "cams.parquet")
    extra = [f"w_{c}" for c in EXTRA]
    res = []
    for y, (cut, end) in SEASONS.items():
        for group in H_GROUPS:
            t0 = time.time()
            kw = HOURLY_VARIANTS[CHOICE[f"{group[0]}-{group[1]}"]]
            cams = cams_all if kw.get("use_cams") else None
            X, yy, w, Xv, meta = split(group, pd.Timestamp(cut), end, wf_fc, cams)
            base = [c for c in X.columns if X[c].notna().any() and c not in extra]
            out = meta.assign(season=y, group=f"{group[0]}-{group[1]}")
            todo = {k: base + [f"w_{c}" for c in v] for k, v in variants.items()} | ({} if into is not None else {"current": base})
            for name, cols in todo.items():
                preds = [_gbm(HOURLY_PARAMS, kw.get("rounds", 300), seed, X[cols], yy, w, "lgbm", kw.get("lgbm_extra"))
                         .predict(Xv[cols].values) + Xv["lag0"].values for seed in SEEDS]
                out[name] = aqi.pm_to_aqi(np.clip(np.expm1(np.mean(preds, axis=0)), 0, None))
            if into is not None:  # add the columns to the stored run (same rows: the splits are deterministic)
                old = into[(into["season"] == y) & (into["group"] == out["group"].iloc[0])]
                assert (old["origin"].values == out["origin"].values).all() and (old["h"].values == out["h"].values).all()
                out = old.assign(**{k: out[k].values for k in variants})
            res.append(out)
            maes = {k: np.abs(out[k] - out["actual"]).mean() for k in ("current", *variants)}
            print(f"{y} {group}: " + "  ".join(f"{k} {v:.2f}" for k, v in maes.items())
                  + f"  ({len(out)} rows, {time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(PROC / "west_test.parquet")
    return df


def summarise(df: pd.DataFrame) -> dict:
    bands = {"1-24": (1, 24), "25-72": (25, 72), "73-168": (73, 168), "1-168": (1, 168)}
    mae = lambda d, k: float(np.abs(d[k] - d["actual"]).mean())  # noqa: E731
    sel = df[df["season"] == 2024]
    chosen = min(VARIANTS, key=lambda k: mae(sel, k))
    out = {"mae": {y: {k: {b: round(mae(df[(df["season"] == y) & df["h"].between(*r)], k), 2) for b, r in bands.items()}
                       for k in ("current", *VARIANTS)} for y in SEASONS},
           "chosen_on_2024": chosen, "chosen_beats_current_2024": mae(sel, chosen) < mae(sel, "current"),
           "diff_vs_current": {y: {k: {b: [round(x, 2) for x in boot(df[(df["season"] == y) & df["h"].between(*r)], "current", k)]
                                       for b, r in bands.items()} for k in (*VARIANTS, *[p for p in POSTHOC if p in df])}
                               for y in SEASONS},
           "rows": {y: int((df["season"] == y).sum()) for y in SEASONS}}
    print(json.dumps(out, indent=1))
    (PROC / "west_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if "--posthoc" in sys.argv:
        df = run(POSTHOC, pd.read_parquet(PROC / "west_test.parquet"))
    else:
        df = pd.read_parquet(PROC / "west_test.parquet") if "--summary" in sys.argv else run()
    summarise(df)
