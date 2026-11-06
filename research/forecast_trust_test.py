"""Does LightGBM trust the weather forecast for the far days too much, because of how it is trained?
(owner, 2026-11-05: idea 2 after the missed week 28.10–3.11)

Two ways the training rows know the future weather better than a real forecast does:

  1. the ventilation and wash-out inputs (`vent_o_T`, `rain_o_T`: mean wind and rain from the origin
     to the target hour, aqf/features.hourly_rows) are summed over the weather frame of lead day 1
     at every horizon. For a target 5 days ahead the hours of day 3–5 then come from forecasts issued
     2–4 days after the origin. Found 2026-11-05. In production the issued forecast is used, so only
     training (and the backtest) is affected; the network (aqf/nn.py) sums lead-matched frames.
     Before the run: on 2025/26 origins the lead-matched mean wind differs from the old one by
     -0.10 ± 0.30 km/h at 120–168 h (0 up to 24 h by construction) — a small difference.
  2. before the archive of forecasts (19.01.2024) every lead gets ERA5 — the weather as it was —
     which is most of the training rows.

Variants (all LightGBM, the other inputs unchanged):
  vent      — 1 fixed: each hour o+k summed from the frame of its own lead day ((k-1)//24 + 1)
  aug       — vent + before the forecast archive, ERA5's surface weather plus the archived forecast
              error of the same lead (forecast minus ERA5), transplanted in 7-day blocks from donor
              days of the nearest calendar month; the donors only before the season's cut-off, so
              nothing of the scored winter enters; the same donor dates for all leads; clipped to
              physical ranges. The upper air (inversions) stays as it is: there is no archive of its
              forecasts to take errors from. Before the run: the transplanted wind error has the
              real spread (sd 2.95 / 3.26 / 3.77 km/h at lead 1 / 5 / 7 against 2.90 / 3.32 / 3.67 in
              Oct–Mar 2024–26); its mean follows the donors' months (Jan–Sep 2024 for the 2024/25 cut);
              daily autocorrelation 0.16 (2024/25 cut) / 0.35 (October 2026 cut) against 0.37 real.
  aug_wind  — vent + the same, for wind speed, direction and gusts only

Rules fixed before the run:
  * head-to-head as in the earlier input tests: trained once on everything before 1 October,
    forecasts every 3 h through the season, all 14 horizon groups with their production settings,
    2 seeds averaged; the current model is LightGBM of model version 2026-10-28.
  * every model is scored on validation rows with the lead-matched ventilation (what production
    gives it); `current_leaky` (the current model on rows built the old way, as in every earlier
    test and the backtest) is reported to show how much the old rows flattered it.
  * choice on winter 2024/25 only: the variant with the lowest MAE over all rows; it passes only if
    it beats `current` there and on both 2025/26 and October 2026 with the 95% bootstrap interval
    over days of the MAE difference entirely below zero. A pass goes on to the full protocol
    (aqf/backtest.py) and is a new model version.

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \\
        PYTHONPATH=. .venv/bin/python research/forecast_trust_test.py [run|summary]

Writes data/processed/forecast_trust_test.parquet (every forecast row) and forecast_trust_test.json.
Trains and writes no production model.

Result (2026-11-05), MAE change vs `current` over 1–168 h (95% interval over days):
                   2024/25 (choice)       2025/26                Oct 2026 (to 3.11)
  current_leaky    -0.13 (-0.19..-0.07)   -0.06 (-0.23..+0.12)   -0.29 (-0.74..+0.04)
  vent             +0.05 (-0.01..+0.11)   +0.23 (+0.11..+0.34)   -0.28 (-0.71..+0.10)   <- chosen, not better
  aug              +0.13 (-0.20..+0.44)   +0.68 (+0.35..+0.99)   -0.91 (-1.33..-0.49)
  aug_wind         +0.39 (+0.07..+0.73)   -0.01 (-0.19..+0.16)   -0.48 (-0.96..-0.04)
All rejected by the rule. The lead-1 ventilation only flatters the scoring a little: the old rows
make the same model look 0.13 AQI better on 2024/25 (0.21 at 73–168 h), 0.06 on 2025/26, 0.29 in
October 2026 (all within ~1%); training on lead-matched rows does not help (worse on 2025/26).
The forecast-error transplant is the best of all in October 2026 (-0.91; issued 27.10–2.11: MAE
36.9 -> 36.0, bias -13.4 -> -9.5) but worse on 2025/26 (+0.68) and no better on 2024/25, so it
is not a dependable gain. On hours with AQI > 150 it changes nothing (2025/26 -38.2 -> -38.4).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import PROC, RAW, load_prev_runs_ensemble, load_target, load_weather, splice_forecast_weather
from aqf.features import Climatology, hourly_rows, pm_state, same_hour_mean, target_times, weather_features
from aqf.models import H_GROUPS, HOURLY_PARAMS, HOURLY_VARIANTS, _gbm, _weights
from research.no2_input_test import boot

SEASONS = {2024: ("2024-10-01", "2025-04-01"), 2025: ("2025-10-01", "2026-04-01"), 2026: ("2026-10-01", None)}
CHOICE = json.loads((PROC / "horizon_choice.json").read_text())
SEEDS = (0, 1)
VARIANTS = ("vent", "aug", "aug_wind")
WIND = ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
BLOCK = 7  # days: keeps the error lasting from day to day (real daily autocorrelation ~0.4)
CLIP = (("wind_speed_10m", 0, None), ("wind_gusts_10m", 0, None), ("precipitation", 0, None),
        ("snowfall", 0, None), ("shortwave_radiation", 0, None),
        ("relative_humidity_2m", 0, 100), ("cloud_cover", 0, 100))


def spliced(archive: pd.DataFrame, prev: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """aqf.data.splice_forecast_weather before weather_features."""
    frames = {}
    for d, f in prev.items():
        w = archive.astype(float).copy()
        common, cols = w.index.intersection(f.index), list(f.columns)
        w.loc[common, cols] = f.loc[common, cols].astype(float).fillna(w.loc[common, cols]).values
        frames[d] = w
    return frames


def errors(archive: pd.DataFrame, prev: dict[int, pd.DataFrame], cut: pd.Timestamp) -> dict[int, pd.DataFrame]:
    """{lead: forecast minus ERA5, hourly}, before `cut` only."""
    errs = {}
    for d, f in prev.items():
        f = f.astype(float)
        a = archive[list(f.columns)].astype(float).reindex(f.index)
        e = f - a
        e["wind_direction_10m"] = (f["wind_direction_10m"] - a["wind_direction_10m"] + 180) % 360 - 180
        errs[d] = e[e.index < cut]
    return errs


def archive_start(prev: dict[int, pd.DataFrame]) -> pd.Timestamp:
    """First full day on which every lead has an archived forecast."""
    return max(f["wind_speed_10m"].first_valid_index() for f in prev.values()).normalize() + pd.Timedelta(days=1)


def augment(raw: dict[int, pd.DataFrame], prev: dict[int, pd.DataFrame], archive: pd.DataFrame,
            cut: pd.Timestamp, cols: list[str], seed: int = 0) -> dict[int, pd.DataFrame]:
    E = errors(archive, prev, cut)
    start = archive_start(prev)
    full = pd.Series(True, index=E[1].index.normalize().unique())
    for e in E.values():
        n = e["wind_speed_10m"].notna().groupby(e.index.normalize()).sum()
        full &= (n.reindex(full.index) == 24).fillna(False)
    pool = full.index[full.values]
    pool_set = set(pool)
    starts = pd.DatetimeIndex([p for p in pool if all(p + pd.Timedelta(days=i) in pool_set for i in range(BLOCK))])
    months = starts.month.values
    rng = np.random.default_rng(seed)
    first = raw[1].index.min().normalize()
    tgt, don = [], []
    for D in pd.date_range(first, start - pd.Timedelta(days=1), freq=f"{BLOCK}D"):
        dist = np.minimum((months - D.month) % 12, (D.month - months) % 12)
        cand = starts[dist == dist.min()]
        Dp = cand[rng.integers(len(cand))]
        n = min(BLOCK * 24, int((start - D) / pd.Timedelta(hours=1)))
        tgt.append(D + pd.to_timedelta(np.arange(n), unit="h"))
        don.append(Dp + pd.to_timedelta(np.arange(n), unit="h"))
    tgt, don = tgt[0].append(tgt[1:]), don[0].append(don[1:])
    frames = {}
    for d, w in raw.items():
        w = w.copy()
        keep = tgt.isin(w.index)
        t = tgt[keep]
        v = w.loc[t, cols] + E[d].reindex(don[keep])[cols].fillna(0.0).values
        for c, lo, hi in CLIP:
            if c in cols:
                v[c] = v[c].clip(lo, hi)
        if "wind_direction_10m" in cols:
            v["wind_direction_10m"] = v["wind_direction_10m"] % 360
        w.loc[t, cols] = v.values
        frames[d] = w
    return frames


def vent_rain(o: pd.DatetimeIndex, h: np.ndarray, feats: dict[int, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Mean wind and rain sum from origin to target, each hour o+k from the frame of lead (k-1)//24+1."""
    ws, pr = np.zeros(len(o)), np.zeros(len(o))
    for L, f in feats.items():
        s = 24 * (L - 1)
        e = np.clip(h, s, 24 * L)
        m = e > s
        if not m.any():
            continue
        cw = f["ws"].fillna(f["ws"].mean()).cumsum()
        cp = f["precip"].fillna(0).cumsum()
        te, ts = o[m] + pd.to_timedelta(e[m], unit="h"), o[m] + pd.Timedelta(hours=s)
        ws[m] += cw.reindex(te).values - cw.reindex(ts).values
        pr[m] += cp.reindex(te).values - cp.reindex(ts).values
    return ws / h, pr


def train_rows(group, cut, feats, cams):
    """Training rows before `cut`, as westerly_wind_test.split (same origins and horizons):
    the rows built the old way and the same rows with the lead-matched ventilation."""
    lo, hi = group
    tgt, _ = load_target(profile_cutoff=cut)
    lp = np.log1p(tgt["pm"])
    lp = lp[lp.index < cut]
    clim = Climatology().fit(lp)
    state, sh7 = pm_state(lp), same_hour_mean(lp)
    rng = np.random.default_rng(0)
    origins = state.index[(state.index.hour % 2 == 0) & state["lag0"].notna()]
    k = min(4, hi - lo + 1)
    H = rng.integers(lo, hi + 1, size=(len(origins), k)) if hi > lo else np.full((len(origins), 1), lo)
    X = hourly_rows(origins, H, lp, state, sh7, clim, feats, tgt["src"], loo=True, cams=cams)
    Xl = X.copy()
    Xl["vent_o_T"], Xl["rain_o_T"] = vent_rain(X.index.get_level_values("origin"), X.index.get_level_values("h").values, feats)
    T = target_times(X)
    y = lp.reindex(T).values - X["lag0"].values
    ok = ~np.isnan(y)
    w = np.asarray(_weights(T[ok], X["src"].values[ok]), float)
    return X[ok].reset_index(drop=True), Xl[ok].reset_index(drop=True), y[ok], w, clim


def val_rows(group, cut, end, feats, cams, clim):
    lo, hi = group
    tgt, _ = load_target(profile_cutoff=cut)
    pm = tgt["pm"]
    lp_all = np.log1p(pm)
    end = min(pd.Timestamp(end) if end else pm.dropna().index.max(), pm.dropna().index.max() - pd.Timedelta(hours=1))
    vo = pd.date_range(cut, end, freq="3h")
    vo = vo[lp_all.ffill(limit=2).reindex(vo).notna().values]
    Xv = hourly_rows(vo, np.arange(lo, hi + 1), lp_all, pm_state(lp_all), same_hour_mean(lp_all), clim, feats, None, cams=cams)
    Xv["src"] = 1.0
    act = aqi.pm_to_aqi(pm.reindex(target_times(Xv)).values)
    okv = ~np.isnan(act) & ~np.isnan(Xv["lag0"].values)
    o, h = Xv.index.get_level_values("origin")[okv], Xv.index.get_level_values("h").values[okv]
    meta = pd.DataFrame({"origin": o, "h": h, "actual": act[okv]})
    leaky = Xv[okv].reset_index(drop=True)
    honest = leaky.copy()
    honest["vent_o_T"], honest["rain_o_T"] = vent_rain(o, h, feats)
    first = h <= 24  # lead 1 only: both ways must agree
    assert np.allclose(honest.loc[first, "vent_o_T"], leaky.loc[first, "vent_o_T"], equal_nan=True)
    return leaky, honest, meta


def run() -> pd.DataFrame:
    archive, prev = load_weather(), load_prev_runs_ensemble()
    raw = spliced(archive, prev)
    base = {d: weather_features(w) for d, w in raw.items()}
    ref = splice_forecast_weather(archive, prev)
    assert all(np.allclose(base[d].values, ref[d].values, equal_nan=True) for d in base), "splice differs"
    cams_all = pd.read_parquet(RAW / "weather" / "cams.parquet")
    res = []
    for y, (cut, end) in SEASONS.items():
        cut = pd.Timestamp(cut)
        aug = {name: {d: weather_features(w) for d, w in augment(raw, prev, archive, cut, cols).items()}
               for name, cols in (("aug", list(prev[1].columns)), ("aug_wind", WIND))}
        for group in H_GROUPS:
            t0 = time.time()
            kw = HOURLY_VARIANTS[CHOICE[f"{group[0]}-{group[1]}"]]
            cams = cams_all if kw.get("use_cams") else None
            X, Xvent, yy, w, clim = train_rows(group, cut, base, cams)
            leaky, honest, meta = val_rows(group, cut, end, base, cams, clim)
            cols = [c for c in X.columns if X[c].notna().any()]
            sets = {"current": X, "vent": Xvent}
            for name in ("aug", "aug_wind"):
                _, Xa, ya, _, _ = train_rows(group, cut, aug[name], cams)
                assert len(Xa) == len(X) and np.allclose(ya, yy, equal_nan=True)
                sets[name] = Xa
            out = meta.assign(season=y, group=f"{group[0]}-{group[1]}")
            for name, Xt in sets.items():
                models = [_gbm(HOURLY_PARAMS, kw.get("rounds", 300), seed, Xt[cols], yy, w, "lgbm", kw.get("lgbm_extra"))
                          for seed in SEEDS]

                def pred(V: pd.DataFrame) -> np.ndarray:
                    lp = np.mean([m.predict(V[cols].values) for m in models], axis=0) + V["lag0"].values
                    return aqi.pm_to_aqi(np.clip(np.expm1(lp), 0, None))

                out[name] = pred(honest)
                if name == "current":
                    out["current_leaky"] = pred(leaky)
            res.append(out)
            maes = {k: np.abs(out[k] - out["actual"]).mean() for k in ("current_leaky", "current", *VARIANTS)}
            print(f"{y} {group}: " + "  ".join(f"{k} {v:.2f}" for k, v in maes.items())
                  + f"  ({len(out)} rows, {time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(PROC / "forecast_trust_test.parquet")
    return df


def summarise(df: pd.DataFrame) -> dict:
    bands = {"1-24": (1, 24), "25-72": (25, 72), "73-168": (73, 168), "1-168": (1, 168)}
    mae = lambda d, k: float(np.abs(d[k] - d["actual"]).mean())  # noqa: E731
    sel = df[df["season"] == 2024]
    red = df["actual"] > 150
    chosen = min(VARIANTS, key=lambda k: mae(sel, k))
    late = df[(df["season"] == 2026) & (df["origin"] >= "2026-10-27") & (df["origin"] < "2026-11-03")]
    keys = ("current_leaky", "current", *VARIANTS)
    out = {
        "mae": {int(y): {k: {b: round(mae(df[(df["season"] == y) & df["h"].between(*r)], k), 2) for b, r in bands.items()}
                         for k in keys} for y in SEASONS},
        "chosen_on_2024": chosen,
        "diff_vs_current": {int(y): {k: {b: [round(x, 2) for x in boot(df[(df["season"] == y) & df["h"].between(*r)], "current", k)]
                                         for b, r in bands.items()} for k in ("current_leaky", *VARIANTS)} for y in SEASONS},
        "bias_on_red_hours": {int(y): {k: round(float((df.loc[(df["season"] == y) & red, k] - df.loc[(df["season"] == y) & red, "actual"]).mean()), 1)
                                       for k in keys} for y in SEASONS},
        "issued_27_10_to_2_11": {k: {"mae": round(mae(late, k), 1), "bias": round(float((late[k] - late["actual"]).mean()), 1)} for k in keys},
        "rows": {int(y): int((df["season"] == y).sum()) for y in SEASONS},
    }
    d = out["diff_vs_current"]
    out["passes"] = bool(mae(sel, chosen) < mae(sel, "current") and d[2025][chosen]["1-168"][2] < 0 and d[2026][chosen]["1-168"][2] < 0)
    print(json.dumps(out, indent=1))
    (PROC / "forecast_trust_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    summarise(pd.read_parquet(PROC / "forecast_trust_test.parquet") if "summary" in sys.argv[1:] else run())
