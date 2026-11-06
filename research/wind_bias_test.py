"""Would the model gain from knowing (1) how wrong the wind forecast has been lately, (2) the wind
and visibility measured at Tashkent airport?

research/late_october_miss.py (2026-11-04): in 28.10–3.11 2026 the weather forecast was too windy on
every day at 1 and 3 days ahead, the airport measured the calm too (METAR UZTT, 7.4 km/h against
9.0 usual), and the wind forecast's daily error lasts: autocorrelation ~0.4, the error already known
at issue time correlates 0.22–0.29 with the coming one. The owner chose to test both (2026-11-04).

Every input below uses only what is known when the forecast is issued: for a target hour T served
from the lead-day-d weather frame (h ≤ 24 d, so the origin is at T − 24 d or later), the windows end
before T − 24 d; the bias uses the three full days before the day of T − 24 d.

  bias (1)       — the 4-model mean's 10 m wind minus ERA5's, mean over those three days:
     bias1       · of the 1-day forecasts (the latest ones that have been verified)
     bias_d      · of the forecasts at the row's own lead d
     bias_calm   · bias1 + the same for calm hours (wind < 5 km/h) per day
  airport (2)    — METAR UZTT (Iowa Environmental Mesonet, every 30 min, since 1931):
     obs         · airport wind, visibility and calm hours over the 24 h before T − 24 d
     obs_bias    · 1-day forecast wind minus airport wind, mean over the three days
     obs_all     · both

Rules fixed before the run (as for NO2, westerly wind, mixing height, the weather ensemble):
  * head-to-head: trained once on everything before 1 October, forecasts every 3 h through the
    season, all 14 horizon groups with their production settings, 2 seeds averaged; the current
    model is LightGBM of model version 2026-10-28 (the served forecast blends it with the network
    from hour 7; the network is not retrained here).
  * choice on winter 2024/25 only, separately for each experiment: the variant with the lowest MAE
    over all (origin, hour) rows; it passes only if it beats the current model there and then on
    both 2025/26 and October 2026 (95% bootstrap interval over days of the MAE difference).
  * the forecast bias exists only where archived forecasts do (2024-01 on); before that it is
    missing, as at the start of real use it would not be.

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \\
        PYTHONPATH=. .venv/bin/python research/wind_bias_test.py [run|summary]

Writes data/processed/wind_bias_test.parquet (every forecast row) and wind_bias_test.json.

Result (2026-11-04), MAE change vs the current model over 1–168 h (95% interval over days):
                 2024/25 (choice)       2025/26                Oct 2026 (to 3.11)
  bias1          +0.21 (-0.00..+0.41)   +0.38 (+0.07..+0.72)   +0.08 (-0.14..+0.30)   <- chosen, worse there
  bias_d         +0.23 (+0.01..+0.45)   +0.10                  +0.19
  bias_calm      +0.29 (+0.06..+0.51)   +0.42                  +0.14
  obs            -0.04 (-0.14..+0.07)   +0.18 (+0.07..+0.28)   -0.12 (-0.28..+0.01)   <- chosen
  obs_bias       +0.52 (+0.24..+0.80)   +0.20                  +0.10
  obs_all        +0.38 (+0.12..+0.65)   +0.21 (-0.02..+0.45)   -0.29 (-0.56..-0.05)
Both rejected by the rule: the wind bias does not beat the current model even on the choice winter,
the airport wind and visibility are a wash there and worse on 2025/26. On hours with AQI > 150 the
bias inputs make the under-forecast worse (-34 -> -38 on 2024/25), the airport ones leave it (-33).
The error does last from day to day, but the model already sees its trace in the measured pollution
of the last days; the bias exists only since 2024 (2.7 winters) to learn from.
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
from research.no2_input_test import boot
from research.westerly_wind_test import split

METAR = RAW / "weather" / "metar_UZTT_full.csv"  # IEM asos.py, UZTT, 2017-12 .. 2026-11-04
SEASONS = {2024: ("2024-10-01", "2025-04-01"), 2025: ("2025-10-01", "2026-04-01"), 2026: ("2026-10-01", None)}
CHOICE = json.loads((PROC / "horizon_choice.json").read_text())
SEEDS = (0, 1)
EXPERIMENTS = {
    "bias": {"bias1": ["x_wsb1"], "bias_d": ["x_wsbd"], "bias_calm": ["x_wsb1", "x_calmb1"]},
    "airport": {"obs": ["x_mws24", "x_mvis24", "x_mcalm24"], "obs_bias": ["x_mwsb1"],
                "obs_all": ["x_mws24", "x_mvis24", "x_mcalm24", "x_mwsb1"]},
}
EXTRA = sorted({c for e in EXPERIMENTS.values() for v in e.values() for c in v})


def airport() -> pd.DataFrame:
    m = pd.read_csv(METAR, na_values="M")
    m = m.set_index(pd.to_datetime(m["valid"]))
    h = pd.DataFrame({"ws": m["sknt"] * 1.852, "vis": m["vsby"] * 1.609}).resample("h").mean()
    return h.asfreq("h")


def _three_days(daily: pd.Series) -> pd.Series:
    """Mean over the three days before each day (at least two present)."""
    return daily.rolling(3, min_periods=2).mean().shift(1)


def extras(d: int, idx: pd.DatetimeIndex, prev: dict[int, pd.DataFrame], era: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:
    """The candidate inputs for the rows served from lead day d, per target hour in `idx`."""
    D0 = (idx - pd.Timedelta(days=d)).normalize()
    end = idx - pd.Timedelta(days=d) - pd.Timedelta(hours=1)  # the last full hour before T − 24 d
    act = era["wind_speed_10m"].astype(float)

    def bias(f: pd.DataFrame, truth: pd.Series) -> pd.Series:
        e = (f["wind_speed_10m"].astype(float) - truth.reindex(f.index)).resample("D").mean()
        return pd.Series(_three_days(e).reindex(D0).values, index=idx)

    calm = lambda s: (s < 5).astype(float).where(s.notna()).resample("D").sum(min_count=12)  # noqa: E731
    out = pd.DataFrame(index=idx)
    out["x_wsb1"] = bias(prev[1], act)
    out["x_wsbd"] = bias(prev[d], act)
    cb = calm(prev[1]["wind_speed_10m"].astype(float)) - calm(act).reindex(calm(prev[1]["wind_speed_10m"].astype(float)).index)
    out["x_calmb1"] = _three_days(cb).reindex(D0).values
    r = met.rolling(24, min_periods=12)
    out["x_mws24"] = r["ws"].mean().reindex(end).values
    out["x_mvis24"] = r["vis"].mean().reindex(end).values
    out["x_mcalm24"] = (met["ws"] < 5).astype(float).where(met["ws"].notna()).rolling(24, min_periods=12).sum().reindex(end).values
    out["x_mwsb1"] = bias(prev[1], met["ws"])
    return out


def weather_by_lead() -> dict[int, pd.DataFrame]:
    """aqf.data.splice_forecast_weather plus the candidate columns."""
    archive, prev, met = load_weather(), load_prev_runs_ensemble(), airport()
    out = {}
    for d, f in prev.items():
        w = archive.astype(float)
        common, cols = w.index.intersection(f.index), list(f.columns)
        w.loc[common, cols] = f.loc[common, cols].astype(float).fillna(w.loc[common, cols]).values
        wf = weather_features(w)
        out[d] = wf.join(extras(d, wf.index, prev, archive, met))
    return out


def run() -> pd.DataFrame:
    wf_fc = weather_by_lead()
    cams_all = pd.read_parquet(RAW / "weather" / "cams.parquet")
    extra = [f"w_{c}" for c in EXTRA]
    variants = {k: v for e in EXPERIMENTS.values() for k, v in e.items()}
    res = []
    for y, (cut, end) in SEASONS.items():
        for group in H_GROUPS:
            t0 = time.time()
            kw = HOURLY_VARIANTS[CHOICE[f"{group[0]}-{group[1]}"]]
            cams = cams_all if kw.get("use_cams") else None
            X, yy, w, Xv, meta = split(group, pd.Timestamp(cut), end, wf_fc, cams)
            base = [c for c in X.columns if X[c].notna().any() and c not in extra]
            out = meta.assign(season=y, group=f"{group[0]}-{group[1]}")
            todo = {"current": base} | {k: base + [f"w_{c}" for c in v] for k, v in variants.items()}
            for name, cols in todo.items():
                preds = [_gbm(HOURLY_PARAMS, kw.get("rounds", 300), seed, X[cols], yy, w, "lgbm", kw.get("lgbm_extra"))
                         .predict(Xv[cols].values) + Xv["lag0"].values for seed in SEEDS]
                out[name] = aqi.pm_to_aqi(np.clip(np.expm1(np.mean(preds, axis=0)), 0, None))
            res.append(out)
            maes = {k: np.abs(out[k] - out["actual"]).mean() for k in ("current", *variants)}
            print(f"{y} {group}: " + "  ".join(f"{k} {v:.2f}" for k, v in maes.items())
                  + f"  ({len(out)} rows, {time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(PROC / "wind_bias_test.parquet")
    return df


def summarise(df: pd.DataFrame) -> dict:
    bands = {"1-24": (1, 24), "25-72": (25, 72), "73-168": (73, 168), "1-168": (1, 168)}
    mae = lambda d, k: float(np.abs(d[k] - d["actual"]).mean())  # noqa: E731
    sel = df[df["season"] == 2024]
    red = df["actual"] > 150
    out = {}
    for exp, variants in EXPERIMENTS.items():
        chosen = min(variants, key=lambda k: mae(sel, k))
        out[exp] = {
            "mae": {int(y): {k: {b: round(mae(df[(df["season"] == y) & df["h"].between(*r)], k), 2) for b, r in bands.items()}
                             for k in ("current", *variants)} for y in SEASONS},
            "chosen_on_2024": chosen, "chosen_beats_current_2024": mae(sel, chosen) < mae(sel, "current"),
            "diff_vs_current": {int(y): {k: {b: [round(x, 2) for x in boot(df[(df["season"] == y) & df["h"].between(*r)], "current", k)]
                                             for b, r in bands.items()} for k in variants} for y in SEASONS},
            "bias_on_red_hours": {int(y): {k: round(float((df.loc[(df["season"] == y) & red, k] - df.loc[(df["season"] == y) & red, "actual"]).mean()), 1)
                                           for k in ("current", *variants)} for y in SEASONS},
        }
    out["rows"] = {int(y): int((df["season"] == y).sum()) for y in SEASONS}
    print(json.dumps(out, indent=1))
    (PROC / "wind_bias_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    df = pd.read_parquet(PROC / "wind_bias_test.parquet") if "summary" in sys.argv[1:] else run()
    summarise(df)
