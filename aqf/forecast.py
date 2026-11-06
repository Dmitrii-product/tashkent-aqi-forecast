"""Production run: refresh data, (re)train, issue the three forecasts.

    .venv/bin/python -m aqf.forecast            # writes output/forecast.json + CSVs
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import aqi, safe_io
from .blend import blend_lp, nn_record, to_aqi
from .data import (ENSEMBLE_MODELS, IQAIR_JSON, MODELS_DIR, PREV_RUN, PROC, ROOT, UPPER,
                   WEATHER_ARCHIVE, ensemble_mean, load_prev_runs, load_prev_runs_ensemble,
                   load_target, load_weather, splice_forecast_weather)
from .features import weather_features
from .models import (FIRST_VERSION, GBM_ONLY_VERSION, GBM_VERSION, MODEL_VERSION, MODEL_VERSIONS, LongRange,
                     compose_daily, daily_aqi, make_hourly_model)
from .sources import cams as cams_src
from .sources import openmeteo, tashkentairdata

OUT = ROOT / "output"
LOG = ROOT / "data" / "forecast_log"  # every issued forecast, for scoring in real use
MODELS = PROC / "models.pkl"
MODELS_PREV = PROC / "models_prev.pkl"
INTERVALS = PROC / "intervals.json"
RETRAIN_DAYS = 7
CAMS = ROOT / "data" / "raw" / "weather" / "cams.parquet"
CHOICE = PROC / "horizon_choice.json"
NN_TIMEOUT = 1200  # s; the network trains 3 copies, about a minute (longer on the first run after a reboot)


def cams_now() -> pd.DataFrame:
    """CAMS history plus the current 5-day CAMS forecast (used by the 7–24 h models)."""
    hist = pd.read_parquet(CAMS)
    try:
        fc = cams_src.fetch_forecast(past_days=3, forecast_days=5)
        hist = pd.concat([hist[hist.index < fc.index.min()], fc])
    except Exception as e:  # noqa: BLE001 - fall back to the cached series
        print("WARNING: CAMS forecast unavailable:", type(e).__name__)
    return hist


def refresh_data() -> list[str]:
    """Update cached sources; a failed source keeps its cache and is reported, not fatal."""
    warnings = []
    try:
        tashkentairdata.fetch(IQAIR_JSON)
    except Exception as e:  # noqa: BLE001 - any network/parse failure falls back to cache
        warnings.append(f"tashkentairdata.uz unavailable ({type(e).__name__}), using the saved data")
    try:
        arch = pd.read_parquet(WEATHER_ARCHIVE)
        start = (arch.index.max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        end = (pd.Timestamp.now(tz="Asia/Tashkent").tz_localize(None) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        new = openmeteo.fetch_archive(start, end).dropna(how="all")
        if new.empty:  # an empty reply must not blank out the last days
            raise ValueError("empty archive reply")
        arch = pd.concat([arch[arch.index < new.index.min()], new])
        safe_io.to_parquet(arch, WEATHER_ARCHIVE)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Open-Meteo weather archive not updated ({type(e).__name__})")
    try:  # upper-air archive (inversion strength) and archived forecasts, used for training
        today = pd.Timestamp.now(tz="Asia/Tashkent").tz_localize(None).normalize()
        up = pd.read_parquet(UPPER)
        new = openmeteo.fetch_upper_history(f"{up.index.max() - pd.Timedelta(days=5):%Y-%m-%d}",
                                            f"{today - pd.Timedelta(days=1):%Y-%m-%d}").dropna(how="all")
        if not new.empty:
            safe_io.to_parquet(pd.concat([up[up.index < new.index.min()], new]), UPPER)
        prev = load_prev_runs()
        start = min(f.index.max() for f in prev.values()) - pd.Timedelta(days=10)
        for model in [None] + ENSEMBLE_MODELS:
            fresh = openmeteo.fetch_previous_runs(f"{start:%Y-%m-%d}", f"{today:%Y-%m-%d}", model=model)
            for d, f in fresh.items():
                path = Path(str(PREV_RUN).format(d)) if model is None else MODELS_DIR / f"{model}_day{d}.parquet"
                old = pd.read_parquet(path) if path.exists() else f.iloc[:0]
                f = f.dropna(how="all")
                if len(f):
                    safe_io.to_parquet(pd.concat([old[old.index < f.index.min()], f]), path)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"archive of weather forecasts not updated ({type(e).__name__})")
    try:
        cams_src.update_cache(CAMS)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"CAMS not updated ({type(e).__name__})")
    for w in warnings:
        print("WARNING:", w)
    return warnings


def spliced_weather(origin: pd.Timestamp) -> pd.DataFrame:
    """Reanalysis up to the origin, then the mean of four global weather models (best match,
    GFS, ICON, GEM): in backtests the multi-model mean cut the 7-day error by ~1 AQI."""
    arch = load_weather()
    frames, upper = openmeteo.fetch_forecast_models(["best_match"] + ENSEMBLE_MODELS)
    fc = ensemble_mean(frames).join(upper, how="left")
    past = arch[arch.index <= origin - pd.Timedelta(days=3)]
    return pd.concat([past, fc[fc.index > past.index.max()]]).sort_index()


def train_all(pm: pd.Series, src: pd.Series) -> dict:
    # train on archived forecasts (reanalysis before 2024), as the model is used in production
    wtrain = splice_forecast_weather(load_weather(), load_prev_runs_ensemble())
    t0 = time.time()
    choice = json.loads(CHOICE.read_text())
    models = {
        "hourly": make_hourly_model(choice, pd.read_parquet(CAMS)).fit(pm, src, wtrain),
        "longrange": LongRange().fit(pm),
        "trained_at": pd.Timestamp.now().isoformat(),
        "data_end": str(pm.dropna().index.max()),
        "version": GBM_VERSION,  # the LightGBM models' version: one that only adds the network keeps them
    }
    print(f"trained in {time.time() - t0:.0f}s")
    return models


def load_or_train(pm, src, force: bool = False) -> dict:
    if MODELS.exists() and not force:
        try:
            m = pickle.loads(MODELS.read_bytes())
        except Exception as e:  # noqa: BLE001 - a damaged file: fall back to the previous model
            print("WARNING: models.pkl unreadable:", type(e).__name__, "- using models_prev.pkl")
            m = pickle.loads(MODELS_PREV.read_bytes()) if MODELS_PREV.exists() else None
        # a model of another version (other inputs) is retrained at once, not at the weekly date
        if (m is not None and m.get("version", FIRST_VERSION) == GBM_VERSION
                and pd.Timestamp.now() - pd.Timestamp(m["trained_at"]) < pd.Timedelta(days=RETRAIN_DAYS)):
            return m
    m = train_all(pm, src)
    blob = pickle.dumps(m)  # fully built before anything on disk changes
    if MODELS.exists():  # kept for aqf.attribution: days must be re-run with a model that has not seen them
        MODELS.replace(MODELS_PREV)
    safe_io.write_bytes(MODELS, blob)
    return m


def network(origin: pd.Timestamp, wx: pd.DataFrame) -> tuple[np.ndarray, list[int]] | None:
    """The network's forecast (log PM2.5, hours 1..168, aqf.nn) from its own process, so that this
    one never imports PyTorch; None if it fails or times out (LightGBM then goes out alone)."""
    with tempfile.TemporaryDirectory() as tmp:
        wxp, outp = Path(tmp) / "wx.parquet", Path(tmp) / "nn.json"
        wx[wx.index >= origin - pd.Timedelta(days=5)].to_parquet(wxp)  # as saved with the issued forecast
        try:
            r = subprocess.run([sys.executable, "-W", "ignore", "-m", "aqf.nn", "serve", "--origin", origin.isoformat(),
                                "--wx", str(wxp), "--out", str(outp)], cwd=ROOT, capture_output=True, text=True,
                               timeout=NN_TIMEOUT)
            if r.returncode != 0:
                raise RuntimeError((r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1])
            js = json.loads(outp.read_text())
            lp = np.array(js["nn_lp"], dtype=float)
            if len(lp) != 168 or not np.isfinite(lp).all():
                raise ValueError(f"{len(lp)} hours, finite: {np.isfinite(lp).all()}")
            print(r.stdout.strip())
            return lp, js["seeds"]
        except Exception as e:  # noqa: BLE001 - the forecast must go out anyway
            print("WARNING: network forecast failed:", type(e).__name__, e)
            return None


def _band(pred_aqi: np.ndarray, keys: np.ndarray, table: dict, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Empirical 10–90% band from backtest log-residual quantiles per horizon bucket."""
    lo = np.full(len(pred_aqi), np.nan)
    hi = np.full(len(pred_aqi), np.nan)
    pm = aqi.aqi_to_pm(pred_aqi)
    for b in table.get(prefix, []):
        m = (keys >= b["lo"]) & (keys <= b["hi"])
        lo[m] = aqi.pm_to_aqi(np.expm1(np.log1p(pm[m]) + b["q10"]))
        hi[m] = aqi.pm_to_aqi(np.expm1(np.log1p(pm[m]) + b["q90"]))
    return lo, hi


def run(force_train: bool = False, refresh: bool = True) -> dict:
    warnings = refresh_data() if refresh else []
    tgt, cal = load_target()
    pm, src = tgt["pm"], tgt["src"]
    obs = pm.dropna()
    origin = obs.index.max()
    wx = spliced_weather(origin)
    models = load_or_train(pm, src, force_train)
    wf = weather_features(wx)
    bands = json.loads(INTERVALS.read_text()) if INTERVALS.exists() else {}

    # hourly: one model per horizon group serves both the 24 h and the 7-day product
    cams = None
    if hasattr(models["hourly"], "set_cams"):
        cams = cams_now()
        models["hourly"].set_cams(cams)
    h = models["hourly"].predict_frame(pd.DatetimeIndex([origin]), pm, wf).reset_index()
    h["time"] = h["origin"] + pd.to_timedelta(h["h"], unit="h")
    # since model version 2026-11-04: from hour 7 on the 50/50 blend with the network (aqf.blend)
    gbm_lp, gbm_aqi = h["pred_lp"].values.copy(), h["pred_aqi"].values.copy()
    version, net = MODEL_VERSION, None
    if MODEL_VERSIONS[-1].get("blend") == "nn":
        assert (h["h"].values == np.arange(1, 169)).all()
        net = network(origin, wx)
        if net is None:
            warnings.append("the neural network did not answer: LightGBM-only forecast")
            version = GBM_ONLY_VERSION
        else:
            h["pred_lp"] = blend_lp(gbm_lp, net[0], h["h"].values)
            h["pred_pm"] = np.expm1(h["pred_lp"]).clip(lower=0)
            h["pred_aqi"] = to_aqi(h["pred_lp"].values)
    h["lo"], h["hi"] = _band(h["pred_aqi"].values, h["h"].values, bands, "hourly")

    models["longrange"] = LongRange().fit(pm)  # cheap (seconds): refit on the newest days
    # daily: day 1 = today. Days 1–7 = daily means of observed + forecast hours,
    # days 8–90 = climatology shifted by the decaying recent anomaly
    # the hour labelled 00:00 opens the day (it ends at midnight), as tashkentairdata.uz counts days
    today = origin.normalize()
    d = models["longrange"].predict_frame(pd.DatetimeIndex([today - pd.Timedelta(days=1)]), pm).reset_index()
    d["date"] = d["origin"] + pd.to_timedelta(d["d"], unit="D")
    def day_means(fc_aqi: np.ndarray) -> pd.Series:
        s = pd.concat([pd.Series(aqi.pm_to_aqi(obs[obs.index >= today].values), index=obs[obs.index >= today].index),
                       pd.Series(fc_aqi, index=h["time"].values)])
        s = s[~s.index.duplicated()]
        g = s.groupby(s.index.normalize())
        return g.mean()[g.count() >= 20]

    day_mean, day_gbm = day_means(h["pred_aqi"].values), day_means(gbm_aqi)
    d["longrange_aqi"] = d["pred_aqi"]
    d["hourly_aqi"] = day_mean.reindex(d["date"]).values
    d.loc[d["d"] > 7, "hourly_aqi"] = np.nan
    d["pred_aqi"] = compose_daily(d)
    d["kind"] = np.where(d["hourly_aqi"].notna(), "model", "clim")
    d["lo"], d["hi"] = _band(d["pred_aqi"].values, d["d"].values, bands, "daily")
    # chance of a red (>150) / purple (>200) day: from the hourly model's error spread for
    # the first week, from how often these dates were red in past years afterwards
    lr = models["longrange"]
    dd = d["d"].values
    for thr, col in ((150, "p_red"), (200, "p_purple")):
        # after the first week fade from the forecast's own chance (its level with the usual
        # spread for the month) to how often the date was red before, like the level fades
        clim_p = lr.climate_prob(pd.DatetimeIndex(d["date"]), thr)
        own_p = lr.prob_above(d["pred_aqi"].values, pd.DatetimeIndex(d["date"]).month.values, thr)
        wgt = np.where(dd > 7, np.exp(-(dd - 7) / 4.0), 0.0)
        p = wgt * own_p + (1 - wgt) * clim_p
        for b in bands.get("daily_q", []):
            m = ((d["d"] >= b["lo"]) & (d["d"] <= b["hi"]) & (d["kind"] == "model")).values
            need = np.log(thr) - np.log(d.loc[m, "pred_aqi"].values)
            p[m] = (np.array(b["q"])[None, :] > need[:, None]).mean(axis=1)
        d[col] = p

    try:  # why the forecast is what it is (SHAP by group of inputs), for the week tab
        from .explain import explain
        why = explain(models["hourly"], origin, pm, wf)
        if net is not None:  # the network's shift of each day, so that the parts still add up to the forecast
            for w in why:
                day = pd.Timestamp(w["date"])
                w["nn"] = round(float(day_mean.get(day, np.nan) - day_gbm.get(day, np.nan)), 1)
                w["forecast"] = round(w["forecast"] + w["nn"], 1)
            why = [w for w in why if not np.isnan(w["nn"])]
    except Exception as e:  # noqa: BLE001 - the forecast itself does not depend on it
        print("WARNING: forecast explanation failed:", type(e).__name__, e)
        why = []

    recent = obs[obs.index > origin - pd.Timedelta(days=7)]
    now = pd.Timestamp.now(tz="Asia/Tashkent").tz_localize(None)
    result = {
        "generated_at": pd.Timestamp.now(tz="Asia/Tashkent").isoformat(),
        "origin": origin.isoformat(),
        "data_age_hours": round((now - origin) / pd.Timedelta(hours=1), 1),
        "warnings": warnings,
        "latest_aqi": float(aqi.pm_to_aqi(obs.iloc[-1])),
        "calibration": cal.__dict__,
        "models_trained_at": models["trained_at"],
        "model_version": version,
        "network": net is not None,
        "recent": [{"t": t.isoformat(), "aqi": round(float(aqi.pm_to_aqi(v)))} for t, v in recent.items()],
        "hourly_24h": _records(h.head(24), "time"),
        "hourly_7d": _records(h, "time"),
        "daily_90d": _records(d, "date", extra=("clim_aqi", "kind", "p_red", "p_purple")),
        "months": month_history(pm),
        "explain": why,
    }
    OUT.mkdir(exist_ok=True)
    safe_io.write_text(OUT / "forecast.json", json.dumps(result, ensure_ascii=False, indent=1))
    LOG.mkdir(parents=True, exist_ok=True)
    issued = LOG / f"{origin:%Y%m%d%H}.json"
    if not issued.exists():  # a re-run from the same data (newer weather) must not replace what was issued
        rec = {k: result[k] for k in ("origin", "generated_at", "model_version", "hourly_7d", "daily_90d")}
        if net is not None:  # LightGBM's own forecast next to the blend, for scoring both and the error breakdown
            rec["hourly_7d_gbm"] = [{"t": pd.Timestamp(t).isoformat(), "aqi": round(float(a))} for t, a in zip(h["time"], gbm_aqi)]
            rec["daily_gbm"] = {f"{k:%Y-%m-%d}": round(float(v)) for k, v in day_gbm.items() if k <= today + pd.Timedelta(days=6)}
        issued.write_text(json.dumps(rec, ensure_ascii=False))
        if net is not None:  # the network part, as aqf.nn shadow would write it
            safe_io.write_text(LOG / f"{origin:%Y%m%d%H}_nn.json", json.dumps(nn_record(
                result["origin"], result["generated_at"], pd.Timestamp.now(tz="Asia/Tashkent").isoformat(),
                net[0], gbm_lp, False, net[1]), ensure_ascii=False))
        # the weather and CAMS this forecast used (5 / 2 days back cover the feature windows), so
        # that aqf.attribution can re-run it exactly and tell its weather error from the rest
        wx[wx.index >= origin - pd.Timedelta(days=5)].to_parquet(LOG / f"{origin:%Y%m%d%H}_wx.parquet")
        if cams is not None:
            cams[cams.index >= origin - pd.Timedelta(days=2)].to_parquet(LOG / f"{origin:%Y%m%d%H}_cams.parquet")
    try:
        from .attribution import update
        update(models)
    except Exception as e:  # noqa: BLE001 - the forecast itself is done; the error breakdown can wait
        print("WARNING: error breakdown not updated:", type(e).__name__, e)
    h.head(24)[["time", "pred_aqi", "lo", "hi"]].round({"pred_aqi": 0, "lo": 0, "hi": 0}).to_csv(OUT / "forecast_24h.csv", index=False)
    h[["time", "pred_aqi", "lo", "hi"]].round({"pred_aqi": 0, "lo": 0, "hi": 0}).to_csv(OUT / "forecast_7d.csv", index=False)
    d[["date", "pred_aqi", "lo", "hi", "clim_aqi", "kind"]].round({"pred_aqi": 0, "lo": 0, "hi": 0, "clim_aqi": 0}).to_csv(OUT / "forecast_90d.csv", index=False)
    return result


def month_history(pm: pd.Series) -> dict:
    """Per calendar month: share of red days in each past year (for the 3-month tab)."""
    a = daily_aqi(pm).dropna()
    g = a.groupby([a.index.year, a.index.month])
    share = g.apply(lambda x: float((x > 150.5).mean())).where(g.size() >= 15).dropna()
    out = {}
    for m in range(1, 13):
        s = share[share.index.get_level_values(1) == m]
        if len(s):
            out[str(m)] = {"by_year": {str(y): round(v, 3) for (y, _), v in s.items()},
                           "min": round(float(s.min()), 3), "max": round(float(s.max()), 3),
                           "mean": round(float(s.mean()), 3)}
    return out


def _records(f: pd.DataFrame, tcol: str, extra: tuple = ()) -> list[dict]:
    out = []
    for _, r in f.iterrows():
        rec = {
            "t": pd.Timestamp(r[tcol]).isoformat(),
            "aqi": round(float(r["pred_aqi"])),
            "lo": None if np.isnan(r["lo"]) else round(float(r["lo"])),
            "hi": None if np.isnan(r["hi"]) else round(float(r["hi"])),
        }
        for c in extra:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                rec[c] = None if np.isnan(v) else (round(float(v), 3) if c.startswith("p_") else round(float(v)))
            else:
                rec[c] = v
        out.append(rec)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args()
    r = run(force_train=a.retrain, refresh=not a.no_refresh)
    print("origin", r["origin"], "latest AQI", r["latest_aqi"], "data age, h", r["data_age_hours"])
    print("next 24h:", [x["aqi"] for x in r["hourly_24h"]])
