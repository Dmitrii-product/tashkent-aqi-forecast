"""Would the model gain from the pressure pattern around Tashkent, not just the weather at one point?
(owner, 2026-11-04: idea 1 after the missed week 28.10–3.11)

Every weather input of the model is for the Tashkent grid point. In 28.10–3.11 2026 the weather
models were too windy there every day (research/late_october_miss.py): a 10 m wind in a basin under
an anticyclone is what they forecast worst, while the large-scale pressure field (is the high
holding, is a front coming) is what they forecast best at 3–7 days. Here: sea-level pressure at
four lowland points 400–500 km around the city, from the same 4-model mean of archived forecasts
(Open-Meteo previous runs: best match, GFS, ICON, GEM; 2024 on) and from ERA5 before, exactly as the
model's other weather (aqf.data.splice_forecast_weather).

  points (lat, lon): TAS 41.31 69.28 (the model's own point), N 45.0 69.0 (Kazakh steppe),
  NW 44.8 65.5 (Kyzylorda), W 39.8 64.4 (Bukhara), SW 37.3 67.3 (Termez)

  grad    — the four differences P(point) - P(TAS) at the target hour T, hPa
  grad24  — grad + the change of each difference over the 24 h before T
  geo     — the geostrophic wind of the plane fitted to the five pressures at T (u, v, speed;
            the wind the pressure field drives above the surface layer, free of the terrain)
            + its speed averaged over the 24 h before T

Rules fixed before the run (as for the wind bias, NO2, westerly wind, mixing height):
  * head-to-head: trained once on everything before 1 October, forecasts every 3 h through the
    season, all 14 horizon groups with their production settings, 2 seeds averaged; the current
    model is LightGBM of model version 2026-10-28 (the served forecast blends it with the network
    from hour 7; the network is not retrained here).
  * the inputs at T come from the weather frame of the row's lead day, like every weather input
    (forecast issued d days before T, ERA5 where there is no archived forecast).
  * choice on winter 2024/25 only: the variant with the lowest MAE over all (origin, hour) rows;
    it passes only if it beats the current model there and then on both 2025/26 and October 2026
    (95% bootstrap interval over days of the MAE difference). A pass goes on to the full protocol
    (aqf/backtest.py, monthly retraining, horizon choice) before any model change, as the westerly
    wind did; a pass there is a new model version.

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \\
        PYTHONPATH=. .venv/bin/python research/pressure_gradient_test.py [fetch|run|summary]

Writes data/raw/weather/points/ (pressure archives), data/processed/pressure_test.parquet (every
forecast row) and pressure_test.json.

Before the run: in winter (Nov–Feb 2018–2026, 882 days, ERA5) the daily AQI goes with the pressure
to the north being above Tashkent's (Spearman +0.25 for N and NW, -0.22 for SW) and with an easterly
geostrophic wind (-0.27; the Siberian high); the geostrophic speed itself does not (-0.05).

Result (2026-11-04), MAE change vs the current model over 1–168 h (95% interval over days):
                 2024/25 (choice)       2025/26                Oct 2026 (to 3.11)
  grad           +0.02 (-0.09..+0.14)   +0.04 (-0.12..+0.18)   +0.60 (+0.29..+0.91)
  grad24         +0.11 (-0.02..+0.23)   +0.11 (-0.06..+0.28)   +0.47 (+0.21..+0.72)
  geo            -0.20 (-0.32..-0.07)   -0.11 (-0.24..+0.03)   +0.42 (+0.13..+0.71)   <- chosen
Rejected by the rule: the geostrophic wind wins the choice winter (-0.32 at 25–72 h) and leans the
same way on 2025/26, but its interval there crosses zero and October 2026 is worse. On hours with
AQI > 150 nothing changes (-34.4 -> -34.5 on 2024/25); for forecasts issued 27.10–2.11 MAE 35.5 ->
36.8. The model already has the local pressure and its 24-h change; the pattern around adds little,
and in the missed week the forecast pressure field was not what went wrong.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import ENSEMBLE_MODELS, PROC, RAW, load_prev_runs_ensemble, load_weather
from aqf.features import weather_features
from aqf.models import H_GROUPS, HOURLY_PARAMS, HOURLY_VARIANTS, _gbm
from aqf.sources.openmeteo import TZ, _get
from research.no2_input_test import boot
from research.westerly_wind_test import split

POINTS = {"TAS": (41.3111, 69.2797), "N": (45.0, 69.0), "NW": (44.8, 65.5), "W": (39.8, 64.4), "SW": (37.3, 67.3)}
DIR = RAW / "weather" / "points"
LEADS = range(1, 8)
SEASONS = {2024: ("2024-10-01", "2025-04-01"), 2025: ("2025-10-01", "2026-04-01"), 2026: ("2026-10-01", None)}
CHOICE = json.loads((PROC / "horizon_choice.json").read_text())
SEEDS = (0, 1)
OUTER = ["N", "NW", "W", "SW"]
VARIANTS = {
    "grad": [f"x_dp_{k}" for k in OUTER],
    "grad24": [f"x_dp_{k}" for k in OUTER] + [f"x_ddp_{k}" for k in OUTER],
    "geo": ["x_ug", "x_vg", "x_vgs", "x_vgs24"],
}
EXTRA = sorted({c for v in VARIANTS.values() for c in v})


def fetch() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(TZ).tz_localize(None).normalize()
    for name, (lat, lon) in POINTS.items():
        p = DIR / f"{name}_era5.parquet"
        if not p.exists():
            frames = []
            for y in range(2018, today.year + 1):
                e = min(today, pd.Timestamp(f"{y}-12-31"))
                js = _get("https://archive-api.open-meteo.com/v1/archive", {
                    "latitude": lat, "longitude": lon, "hourly": "pressure_msl",
                    "start_date": f"{y}-01-01", "end_date": f"{e:%Y-%m-%d}", "timezone": TZ})
                frames.append(pd.Series(js["hourly"]["pressure_msl"], index=pd.to_datetime(js["hourly"]["time"])))
                time.sleep(2)
            pd.concat(frames).dropna().rename("pmsl").to_frame().to_parquet(p)
            print(name, "era5 elevation", js.get("elevation"), flush=True)
        for model in [None] + ENSEMBLE_MODELS:
            p = DIR / f"{name}_{model or 'best_match'}_prev.parquet"
            if p.exists():
                continue
            frames = []
            for s in pd.date_range("2024-01-01", today, freq="6MS"):
                e = min(today, s + pd.DateOffset(months=6) - pd.Timedelta(days=1))
                params = {"latitude": lat, "longitude": lon,
                          "hourly": ",".join(f"pressure_msl_previous_day{d}" for d in LEADS),
                          "start_date": f"{s:%Y-%m-%d}", "end_date": f"{e:%Y-%m-%d}", "timezone": TZ}
                if model:
                    params["models"] = model
                js = _get("https://previous-runs-api.open-meteo.com/v1/forecast", params)
                h = js["hourly"]
                frames.append(pd.DataFrame({d: h[f"pressure_msl_previous_day{d}"] for d in LEADS}, index=pd.to_datetime(h["time"])))
                time.sleep(2)
            df = pd.concat(frames)
            df.columns = [f"day{d}" for d in df.columns]
            df.dropna(how="all").to_parquet(p)
            print(name, model or "best_match", len(df), flush=True)


def pressures() -> dict[int, pd.DataFrame]:
    """{lead day: hourly frame of the five pressures}: 4-model forecast mean, ERA5 where none."""
    era = {k: pd.read_parquet(DIR / f"{k}_era5.parquet")["pmsl"] for k in POINTS}
    out = {}
    for d in LEADS:
        cols = {}
        for k in POINTS:
            fc = pd.concat([pd.read_parquet(DIR / f"{k}_{m or 'best_match'}_prev.parquet")[f"day{d}"].astype(float)
                            for m in [None] + ENSEMBLE_MODELS], axis=1).mean(axis=1)
            idx = era[k].index.union(fc.index)
            cols[k] = fc.reindex(idx).fillna(era[k].reindex(idx))
        out[d] = pd.DataFrame(cols).asfreq("h").interpolate(limit=3)
    return out


def geostrophic(P: pd.DataFrame) -> pd.DataFrame:
    """Least-squares plane through the five pressures -> geostrophic wind (km/h)."""
    lat0, lon0 = POINTS["TAS"]
    xy = np.array([[(lon - lon0) * 111.32e3 * np.cos(np.deg2rad(lat0)), (lat - lat0) * 110.57e3] for lat, lon in POINTS.values()])
    A = np.c_[np.ones(len(xy)), xy]
    coef = np.linalg.lstsq(A, (P[list(POINTS)].values * 100).T, rcond=None)[0]  # Pa, Pa/m, Pa/m
    f, rho = 2 * 7.292e-5 * np.sin(np.deg2rad(lat0)), 1.25
    ug, vg = -coef[2] / (rho * f) * 3.6, coef[1] / (rho * f) * 3.6
    return pd.DataFrame({"ug": ug, "vg": vg}, index=P.index)


def extras(P: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=P.index)
    for k in OUTER:
        out[f"x_dp_{k}"] = P[k] - P["TAS"]
        out[f"x_ddp_{k}"] = out[f"x_dp_{k}"] - out[f"x_dp_{k}"].shift(24)
    g = geostrophic(P.dropna())
    out["x_ug"], out["x_vg"] = g["ug"], g["vg"]
    out["x_vgs"] = np.hypot(out["x_ug"], out["x_vg"])
    out["x_vgs24"] = out["x_vgs"].rolling(24, min_periods=12).mean()
    return out


def weather_by_lead() -> dict[int, pd.DataFrame]:
    """aqf.data.splice_forecast_weather plus the candidate columns of the same lead."""
    archive, prev, P = load_weather(), load_prev_runs_ensemble(), pressures()
    out = {}
    for d, f in prev.items():
        w = archive.astype(float)
        common, cols = w.index.intersection(f.index), list(f.columns)
        w.loc[common, cols] = f.loc[common, cols].astype(float).fillna(w.loc[common, cols]).values
        wf = weather_features(w)
        out[d] = wf.join(extras(P[d]).reindex(wf.index))
    return out


def run() -> pd.DataFrame:
    wf_fc = weather_by_lead()
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
    df.to_parquet(PROC / "pressure_test.parquet")
    return df


def summarise(df: pd.DataFrame) -> dict:
    bands = {"1-24": (1, 24), "25-72": (25, 72), "73-168": (73, 168), "1-168": (1, 168)}
    mae = lambda d, k: float(np.abs(d[k] - d["actual"]).mean())  # noqa: E731
    sel = df[df["season"] == 2024]
    red = df["actual"] > 150
    chosen = min(VARIANTS, key=lambda k: mae(sel, k))
    late = df[(df["season"] == 2026) & (df["origin"] >= "2026-10-27") & (df["origin"] < "2026-11-03")]
    out = {
        "mae": {int(y): {k: {b: round(mae(df[(df["season"] == y) & df["h"].between(*r)], k), 2) for b, r in bands.items()}
                         for k in ("current", *VARIANTS)} for y in SEASONS},
        "chosen_on_2024": chosen, "chosen_beats_current_2024": mae(sel, chosen) < mae(sel, "current"),
        "diff_vs_current": {int(y): {k: {b: [round(x, 2) for x in boot(df[(df["season"] == y) & df["h"].between(*r)], "current", k)]
                                         for b, r in bands.items()} for k in VARIANTS} for y in SEASONS},
        "bias_on_red_hours": {int(y): {k: round(float((df.loc[(df["season"] == y) & red, k] - df.loc[(df["season"] == y) & red, "actual"]).mean()), 1)
                                       for k in ("current", *VARIANTS)} for y in SEASONS},
        "issued_27_10_to_2_11": {k: {"mae": round(mae(late, k), 1), "bias": round(float((late[k] - late["actual"]).mean()), 1)}
                                 for k in ("current", *VARIANTS)},
        "rows": {int(y): int((df["season"] == y).sum()) for y in SEASONS},
    }
    print(json.dumps(out, indent=1))
    (PROC / "pressure_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if "fetch" in args:
        fetch()
    elif "summary" in args:
        summarise(pd.read_parquet(PROC / "pressure_test.parquet"))
    else:
        summarise(run())
