"""Would the hourly forecast gain from (a) how much the weather models disagree, (b) more models?

In the committee week (28.10–2.11 2026, docs/checks/2026-10-27-committee-vs-forecast.md) the weather
forecast promised a front day after day that did not come: the error breakdown put -15 AQI at 1 day
and -33 at 3 days on the weather. The owner chose to test two ideas (2026-11-04):

  spread — the spread between the four models whose mean the model gets (best match, GFS, ICON,
           GEM; archived forecasts by lead time since 2024-01), so that it can learn to trust a
           mean of models that disagree less:
             wind      — standard deviation of the 10 m wind speed at the hour and its 24 h mean
             all       — 24 h means of the spread of wind speed, temperature and surface pressure,
                         and the spread of the 24 h rain totals
             wind_all  — both
  models — more models in the mean (Open-Meteo previous runs):
             ifs       — + ECMWF IFS 0.25° (archived since 2024-02-04; no gusts and snowfall)
             aifs      — + ECMWF AIFS, the machine-learning model (archived since 2025-02-18 only)
             ifs_aifs  — both

Rules fixed before the run (as for NO2, westerly wind, mixing height):
  * head-to-head: trained once on everything before 1 October, forecasts every 3 h through the
    season, all 14 horizon groups with their production settings, 2 seeds averaged; the current
    model is model version 2026-10-28 (LightGBM with westerly wind).
  * choice on winter 2024/25 only, separately for each experiment: the variant with the lowest MAE
    over all (origin, hour) rows; it passes only if it beats the current model there and then on
    both 2025/26 and October 2026 (95% bootstrap interval over days of the MAE difference).
  * AIFS is not archived for 2024/25: there "aifs" equals the current model and "ifs_aifs" equals
    "ifs". Its gain is reported on 2025/26 and October 2026 only, as an observation; it cannot pass
    the rule before winter 2026/27 is scored.

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \\
        PYTHONPATH=. .venv/bin/python research/weather_ensemble_test.py [fetch|spread|models|summary]

Writes data/raw/weather/models_research/<model>_day<d>.parquet (the two ECMWF archives),
data/processed/ens_{spread,models}_test.parquet (every forecast row) and ens_test.json (summary).

Result (2026-11-04), MAE change vs the current model over 1–168 h (95% interval over days):
                 2024/25 (choice)       2025/26                Oct 2026 (to 3.11)
  spread wind    +0.14 (-0.09..+0.37)   +0.07 (-0.16..+0.31)   +0.03 (-0.27..+0.36)   <- chosen, worse there
  spread all     +0.34 (+0.04..+0.65)   +0.04                  +0.22
  wind_all       +0.35 (+0.05..+0.66)   -0.06 (-0.29..+0.17)   +0.33
  models ifs     -0.24 (-0.41..-0.08)   +0.32 (+0.15..+0.49)   +0.23 (-0.18..+0.64)
  models aifs    -0.15 (-0.22..-0.09)   +0.20 (+0.01..+0.38)   +0.04 (-0.69..+0.75)
  ifs_aifs       -0.33 (-0.51..-0.15)   +0.28 (+0.03..+0.53)   +0.26 (-0.41..+0.89)   <- chosen
Both rejected by the rule. The spread does not beat the current model even on the choice winter.
More models help on 2024/25 and hurt on 2025/26 — over the week, worst at 3–7 days (IFS +0.54);
best_match is already mostly IFS there (temperature r = 0.98), so the mean leans further on one model.
Note on the rule above: AIFS (archived from 2025-02-18) does enter the last six weeks of 2024/25,
so there "aifs" is not exactly the current model; the choice left it out as planned, and with it
the outcome is the same (every variant is worse on 2025/26).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import (ENSEMBLE_MODELS, MODELS_DIR, PROC, RAW, ensemble_mean, load_prev_runs,
                      load_weather)
from aqf.features import weather_features
from aqf.models import H_GROUPS, HOURLY_PARAMS, HOURLY_VARIANTS, _gbm
from aqf.sources import openmeteo
from research.no2_input_test import boot
from research.westerly_wind_test import split

RDIR = RAW / "weather" / "models_research"
NEW_MODELS = {"ifs": "ecmwf_ifs025", "aifs": "ecmwf_aifs025_single"}
SINCE = {"ecmwf_ifs025": "2024-02-01", "ecmwf_aifs025_single": "2025-02-15"}
SEASONS = {2024: ("2024-10-01", "2025-04-01"), 2025: ("2025-10-01", "2026-04-01"), 2026: ("2026-10-01", None)}
CHOICE = json.loads((PROC / "horizon_choice.json").read_text())
SEEDS = (0, 1)
SPREAD = {
    "wind": ["x_sdws", "x_sdws24"],
    "all": ["x_sdws24", "x_sdt24", "x_sdps24", "x_sdpr24"],
    "wind_all": ["x_sdws", "x_sdws24", "x_sdt24", "x_sdps24", "x_sdpr24"],
}
MODELS = {"ifs": ["ifs"], "aifs": ["aifs"], "ifs_aifs": ["ifs", "aifs"]}


def fetch() -> None:
    RDIR.mkdir(parents=True, exist_ok=True)
    end = f"{pd.Timestamp.now() - pd.Timedelta(days=1):%Y-%m-%d}"
    for m, start in SINCE.items():
        if all((RDIR / f"{m}_day{d}.parquet").exists() for d in range(1, 8)):
            continue
        t0 = time.time()
        runs = openmeteo.fetch_previous_runs(start, end, model=m)
        for d, f in runs.items():
            f.to_parquet(RDIR / f"{m}_day{d}.parquet")
        print(f"{m}: {min(f.index.min() for f in runs.values())} .. {max(f.index.max() for f in runs.values())} ({time.time() - t0:.0f}s)", flush=True)


def per_model(extra: list[str] = ()) -> dict[int, list[pd.DataFrame]]:
    """{lead day: [best match, GFS, ICON, GEM, + extra]} archived forecasts, as aqf.data loads them."""
    base = load_prev_runs()
    out = {}
    for d, f in base.items():
        frames = [f]
        for m in ENSEMBLE_MODELS:
            p = MODELS_DIR / f"{m}_day{d}.parquet"
            if p.exists() and len(g := pd.read_parquet(p)):
                frames.append(g)
        for k in extra:
            p = RDIR / f"{NEW_MODELS[k]}_day{d}.parquet"
            if p.exists() and len(g := pd.read_parquet(p)):
                frames.append(g.reindex(columns=f.columns))
        out[d] = frames
    return out


def spread(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Disagreement between the models, per hour (NaN where fewer than 2 have a value)."""
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.union(f.index)
    fr = [f.reindex(idx) for f in frames]
    sd = lambda c: pd.concat([f[c] for f in fr], axis=1).std(axis=1, ddof=0).where(  # noqa: E731
        pd.concat([f[c] for f in fr], axis=1).notna().sum(axis=1) >= 2)
    s = pd.DataFrame(index=idx)
    s["x_sdws"] = sd("wind_speed_10m")
    s["x_sdws24"] = s["x_sdws"].rolling(24, min_periods=12).mean()
    s["x_sdt24"] = sd("temperature_2m").rolling(24, min_periods=12).mean()
    s["x_sdps24"] = sd("surface_pressure").rolling(24, min_periods=12).mean()
    rain24 = [f["precipitation"].rolling(24, min_periods=12).sum() for f in fr]
    r = pd.concat(rain24, axis=1)
    s["x_sdpr24"] = r.std(axis=1, ddof=0).where(r.notna().sum(axis=1) >= 2)
    return s


def splice(archive: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """aqf.data.splice_forecast_weather for one lead day, before the features."""
    w = archive.astype(float)
    common, cols = w.index.intersection(fc.index), list(fc.columns)
    w.loc[common, cols] = fc.loc[common, cols].astype(float).fillna(w.loc[common, cols]).values
    return w


def wf_spread(archive: pd.DataFrame) -> dict[int, pd.DataFrame]:
    out = {}
    for d, frames in per_model().items():
        out[d] = weather_features(splice(archive, ensemble_mean(frames))).join(spread(frames))
    return out


def wf_models(archive: pd.DataFrame, extra: list[str]) -> dict[int, pd.DataFrame]:
    return {d: weather_features(splice(archive, ensemble_mean(frames))) for d, frames in per_model(extra).items()}


def fit_predict(X, yy, w, Xv, cols, kw) -> np.ndarray:
    preds = [_gbm(HOURLY_PARAMS, kw.get("rounds", 300), seed, X[cols], yy, w, "lgbm", kw.get("lgbm_extra"))
             .predict(Xv[cols].values) + Xv["lag0"].values for seed in SEEDS]
    return aqi.pm_to_aqi(np.clip(np.expm1(np.mean(preds, axis=0)), 0, None))


def run(exp: str) -> pd.DataFrame:
    archive = load_weather()
    cams_all = pd.read_parquet(RAW / "weather" / "cams.parquet")
    weathers = {"current": wf_spread(archive)} if exp == "spread" else \
        {"current": wf_models(archive, [])} | {k: wf_models(archive, v) for k, v in MODELS.items()}
    variants = SPREAD if exp == "spread" else MODELS
    extra = [f"w_{c}" for c in sorted({c for v in SPREAD.values() for c in v})]
    res = []
    for y, (cut, end) in SEASONS.items():
        for group in H_GROUPS:
            t0 = time.time()
            kw = HOURLY_VARIANTS[CHOICE[f"{group[0]}-{group[1]}"]]
            cams = cams_all if kw.get("use_cams") else None
            X, yy, w, Xv, meta = split(group, pd.Timestamp(cut), end, weathers["current"], cams)
            base = [c for c in X.columns if X[c].notna().any() and c not in extra]
            out = meta.assign(season=y, group=f"{group[0]}-{group[1]}")
            out["current"] = fit_predict(X, yy, w, Xv, base, kw)
            for name, v in variants.items():
                if exp == "spread":
                    out[name] = fit_predict(X, yy, w, Xv, base + [f"w_{c}" for c in v], kw)
                else:  # other weather: other rows of features, same (origin, hour) pairs
                    X2, y2, w2, Xv2, m2 = split(group, pd.Timestamp(cut), end, weathers[name], cams)
                    assert (m2["origin"].values == meta["origin"].values).all() and (m2["h"].values == meta["h"].values).all()
                    out[name] = fit_predict(X2, y2, w2, Xv2, base, kw)
            res.append(out)
            maes = {k: np.abs(out[k] - out["actual"]).mean() for k in ("current", *variants)}
            print(f"{exp} {y} {group}: " + "  ".join(f"{k} {v:.2f}" for k, v in maes.items())
                  + f"  ({len(out)} rows, {time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(PROC / f"ens_{exp}_test.parquet")
    return df


def summarise() -> dict:
    bands = {"1-24": (1, 24), "25-72": (25, 72), "73-168": (73, 168), "1-168": (1, 168)}
    mae = lambda d, k: float(np.abs(d[k] - d["actual"]).mean())  # noqa: E731
    out = {}
    for exp, variants in (("spread", SPREAD), ("models", MODELS)):
        p = PROC / f"ens_{exp}_test.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        sel = df[df["season"] == 2024]
        cand = [k for k in variants if k != "aifs"]  # "aifs" is the current model on 2024/25
        chosen = min(cand, key=lambda k: mae(sel, k))
        out[exp] = {
            "mae": {int(y): {k: {b: round(mae(df[(df["season"] == y) & df["h"].between(*r)], k), 2) for b, r in bands.items()}
                             for k in ("current", *variants)} for y in SEASONS},
            "chosen_on_2024": chosen, "chosen_beats_current_2024": mae(sel, chosen) < mae(sel, "current"),
            "diff_vs_current": {int(y): {k: {b: [round(x, 2) for x in boot(df[(df["season"] == y) & df["h"].between(*r)], "current", k)]
                                             for b, r in bands.items()} for k in variants} for y in SEASONS},
            "rows": {int(y): int((df["season"] == y).sum()) for y in SEASONS}}
    print(json.dumps(out, indent=1))
    (PROC / "ens_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    what = sys.argv[1:] or ["fetch", "spread", "models", "summary"]
    if "fetch" in what:
        fetch()
    for exp in ("spread", "models"):
        if exp in what:
            run(exp)
    summarise()
