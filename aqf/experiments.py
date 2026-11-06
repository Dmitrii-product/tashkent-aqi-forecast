"""Head-to-head comparison of hourly-model variants on the same walk-forward backtest.

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
        .venv/bin/python -m aqf.experiments --seasons 2025,2026,2024

Every variant is trained only on data before 1 October of the test season and fed the
archived weather forecast (as in production). Results: data/processed/exp_<season>.parquet
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import aqi
from .backtest import load_prev_runs, season_bounds, splice_forecast_weather
from .data import load_prev_runs_ensemble
from .data import PROC, RAW, load_target, load_weather
from .features import weather_features
from .models import HourlyModel

IQAIR_START = pd.Timestamp("2025-01-05")

VARIANTS = {
    "hgb": dict(algo="hgb"),
    "hgb_hl365": dict(algo="hgb", half_life_days=365),
    "hgb_hl120": dict(algo="hgb", half_life_days=120),
    "hgb_2y": dict(algo="hgb", train_years=2),
    "hgb_iqair": dict(algo="hgb", iqair_only=True),
    "xgb": dict(algo="xgb"),
    "lgbm": dict(algo="lgbm"),
    # learn from the same archived weather forecasts the model is fed in production
    # (2024+; reanalysis before that), so forecast-weather biases are learned per lead
    "hgb_fcwx": dict(algo="hgb", train_fc=True),
    "lgbm_fcwx": dict(algo="lgbm", train_fc=True),
    "lgbm_fcwx_hl365": dict(algo="lgbm", train_fc=True, half_life_days=365),
    "lgbm_cams": dict(algo="lgbm", use_cams=True),
    "lgbm_fcwx_cams": dict(algo="lgbm", train_fc=True, use_cams=True),
    # + inversion strength from upper-air temperatures (near-analysis archive: optimistic
    # beyond ~1 day, since no archive of upper-air forecasts by lead time exists)
    "lgbm_up": dict(algo="lgbm", upper=True),
    "lgbm_fcwx_up": dict(algo="lgbm", train_fc=True, upper=True),
    "lgbm_fcwx_cams_up": dict(algo="lgbm", train_fc=True, use_cams=True, upper=True),
    # --- round 3, all on top of the production setup (lgbm_fcwx_up)
    "v2_l1": dict(algo="lgbm", train_fc=True, upper=True, lgbm_extra=dict(objective="l1")),
    "v2_huber": dict(algo="lgbm", train_fc=True, upper=True, lgbm_extra=dict(objective="huber", alpha=0.3)),
    "v2_big": dict(algo="lgbm", train_fc=True, upper=True, rounds=900,
                   lgbm_extra=dict(learning_rate=0.02, num_leaves=63, min_child_samples=300)),
    "v2_small": dict(algo="lgbm", train_fc=True, upper=True, rounds=600,
                     lgbm_extra=dict(learning_rate=0.03, num_leaves=15, min_child_samples=1000)),
    "v2_more_rows": dict(algo="lgbm", train_fc=True, upper=True, origin_step=1, per_origin=8),
    "v2_clean": dict(algo="lgbm", train_fc=True, upper=True, clean_target=True),
    # multi-model mean of archived weather forecasts (best match + GFS + ICON + GEM)
    "v2_mm": dict(algo="lgbm", train_fc=True, upper=True, ensemble=True),
    # round 4: candidates for per-horizon selection (all on ensemble weather)
    "v3_mm_cams": dict(algo="lgbm", train_fc=True, upper=True, ensemble=True, use_cams=True),
    "v3_mm_hgb": dict(algo="hgb", train_fc=True, upper=True, ensemble=True),
    "v3_mm_l1": dict(algo="lgbm", train_fc=True, upper=True, ensemble=True, lgbm_extra=dict(objective="l1")),
    "v3_mm_big": dict(algo="lgbm", train_fc=True, upper=True, ensemble=True, rounds=900,
                      lgbm_extra=dict(learning_rate=0.02, num_leaves=63, min_child_samples=300)),
    # other algorithm families
    "v3_mm_ridge": dict(algo="ridge", train_fc=True, upper=True, ensemble=True),
    "v3_mm_et": dict(algo="et", train_fc=True, upper=True, ensemble=True),
    "v3_mm_mlp": dict(algo="mlp", train_fc=True, upper=True, ensemble=True),
}


def analog_year(lp: pd.Series, origins: pd.DatetimeIndex, H: np.ndarray, trend_days: int,
                smooth: int) -> pd.Series:
    """Last year's hourly curve (same weekday, 364 days back) shifted by how much the last
    `trend_days` days differ from the same days a year earlier — the 'last year's dynamics
    plus this year's trend' baseline."""
    ly = lp.rolling(smooth, center=True, min_periods=1).mean() if smooth > 1 else lp
    m = lp.rolling(24 * trend_days, min_periods=12 * trend_days).mean()
    lag = pd.Timedelta(days=364)
    o = pd.DatetimeIndex(np.repeat(origins.values, len(H)))
    h = np.tile(H, len(origins))
    T = o + pd.to_timedelta(h, unit="h")
    trend = m.reindex(o).values - m.reindex(o - lag).values
    pred = ly.reindex(T - lag).values + trend
    return pd.Series(aqi.pm_to_aqi(np.expm1(pred)),
                     index=pd.MultiIndex.from_arrays([o, h], names=["origin", "h"]))


def run_season(y: int, variants: dict, step: int = 6) -> pd.DataFrame:
    start, end = season_bounds(y)
    tgt, _ = load_target(profile_cutoff=start)
    pm, src = tgt["pm"], tgt["src"]
    end = min(end, pm.dropna().index.max() - pd.Timedelta(hours=1))
    wx = load_weather(upper=False)
    prev = load_prev_runs()
    wf = weather_features(wx)
    wf_fc = splice_forecast_weather(wx, prev)
    needs_up = any(v.get("upper") for v in variants.values())
    if needs_up:
        wx_up = load_weather(upper=True)
        wf_up, wf_fc_up = weather_features(wx_up), splice_forecast_weather(wx_up, prev)
    if any(v.get("ensemble") for v in variants.values()):
        wf_fc_mm = splice_forecast_weather(wx_up, load_prev_runs_ensemble())
    lp = np.log1p(pm)
    origins = pd.date_range(start, end, freq=f"{step}h")
    origins = origins[lp.ffill(limit=2).reindex(origins).notna().values]
    H = np.arange(1, 169)
    out = None
    for name, kw in variants.items():
        kw = dict(kw)
        if kw.pop("iqair_only", False):
            if start - IQAIR_START < pd.Timedelta(days=180):
                continue
            kw["train_start"] = IQAIR_START
        if (yrs := kw.pop("train_years", None)):
            kw["train_start"] = start - pd.Timedelta(days=365 * yrs)
        up = kw.pop("upper", False)
        wtrain = (wf_fc_up if up else wf_fc) if kw.pop("train_fc", False) else (wf_up if up else wf)
        wpred = wf_fc_up if up else wf_fc
        if kw.pop("ensemble", False):
            wtrain = wpred = wf_fc_mm
        if kw.pop("use_cams", False):
            kw["cams"] = pd.read_parquet(RAW / "weather" / "cams.parquet")
        t0 = time.time()
        m = HourlyModel(max_h=168, **kw).fit(pm, src, wtrain, cutoff=start)
        p = m.predict_frame(origins, pm, wpred)
        if out is None:
            out = p[["persist_aqi", "clim_aqi", "same_hour_aqi"]].copy()
        out[f"pred_{name}"] = p["pred_aqi"]
        print(f"  {y}/{y + 1} {name}: {time.time() - t0:.0f}s, rows {m.n_train}", flush=True)
    for td in (7, 30):
        for sm in (1, 3):
            out[f"analog_t{td}_s{sm}"] = analog_year(lp, origins, H, td, sm).reindex(out.index).values
    out = out.reset_index()
    out["T"] = out["origin"] + pd.to_timedelta(out["h"], unit="h")
    out["actual_aqi"] = aqi.pm_to_aqi(pm.reindex(out["T"]).values)
    out["season"] = y
    return out.dropna(subset=["actual_aqi"])


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """MAE / MAPE per method on the rows every method could score (same sample for all)."""
    cols = [c for c in df.columns if c.startswith(("pred_", "analog_")) or c in ("persist_aqi", "clim_aqi")]
    df = df.dropna(subset=cols + ["actual_aqi"])
    rows = {}
    for lab, m in (("1-24h", df["h"] <= 24), ("1-168h", df["h"] <= 168), ("25-72h", (df["h"] > 24) & (df["h"] <= 72)),
                   ("73-168h", df["h"] > 72)):
        s = df[m]
        a = s["actual_aqi"].values
        rows[f"MAE {lab}"] = {c: np.nanmean(np.abs(s[c].values - a)) for c in cols}
        rows[f"MAPE% {lab}"] = {c: 100 * np.nanmean(np.abs(s[c].values - a) / a) for c in cols}
    return pd.DataFrame(rows).round(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2025,2026,2024")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    variants = {k: VARIANTS[k] for k in a.variants.split(",")}
    for y in [int(s) for s in a.seasons.split(",")]:
        df = run_season(y, variants)
        df.to_parquet(PROC / f"exp_{y}{a.tag}.parquet")
        print(summary(df).to_string(), flush=True)
