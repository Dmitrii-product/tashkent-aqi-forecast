"""Does yesterday's satellite NO2 (Sentinel-5P) make the hourly forecast better?

Rules fixed before the run (owner's request 2026-10-27, protocol as for every model change):
  * what the model may know: NO2 of the passes up to the day before the origin (the pass is
    about 13:00 local; the quick product NRTI is in the bucket by ~16:00, so the 06:00 forecast
    sees yesterday's pass). The history uses the final product OFFL; production would use NRTI.
  * variants on top of the current model, each with the production settings of its horizon group:
      last   — last cloud-free pass within 7 days (µmol/m²) and its age in days
      week   — last + mean of the cloud-free passes of the last 7 days
      ring   — week + the same two for the ring 30–50 km out (background of the region)
  * head-to-head as in aqf.heating / aqf.tune: trained once on everything before 1 October,
    forecasts every 3 h through the season with archived forecast weather, all 14 horizon groups,
    2 seeds averaged.
  * choice on winter 2024/25 only: the variant with the lowest MAE over all (origin, hour) rows.
    Also a per-group router: NO2 only in the groups where the chosen variant beat the current one
    on 2024/25.
  * accepted only if better on both 2025/26 and October 2026, by more than the noise (95%
    bootstrap interval over days of the MAE difference, seeds averaged).

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
        PYTHONPATH=. .venv/bin/python research/no2_input_test.py      # ~1.5 h

Writes data/processed/no2_test.parquet (every forecast row) and no2_test.json (summary).

Result (2026-10-27): rejected. MAE change vs the current model over 1–168 h (95% interval):
  2024/25 (choice)  last +0.09 (-0.00..+0.19), week +0.24, ring +0.30 -> no variant beats current
  2025/26           last -0.18 (-0.29..-0.08), week -0.23, ring -0.19
  Oct 2026          last -0.11 (-0.38..+0.16), week -0.32, ring -0.34
against MAE 26–33 AQI (under 1%). Best at 25–72 h on the unseen periods (-0.3, -0.6) but +0.16
there on 2024/25; the per-group router (2, 3, 4–6, 13–18, 145–168 h) gives -0.05 on both.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import PROC, RAW, load_prev_runs_ensemble, load_target, load_weather, splice_forecast_weather
from aqf.features import Climatology, hourly_rows, pm_state, same_hour_mean, target_times
from aqf.models import H_GROUPS, HOURLY_PARAMS, HOURLY_VARIANTS, _gbm, _weights
from aqf.satellite import no2_series

SEASONS = {2024: ("2024-10-01", "2025-04-01"), 2025: ("2025-10-01", "2026-04-01"), 2026: ("2026-10-01", None)}
CHOICE = json.loads((PROC / "horizon_choice.json").read_text())
SEEDS = (0, 1)
WINDOW = 7
VARIANTS = {
    "last": ["no2_last", "no2_age"],
    "week": ["no2_last", "no2_age", "no2_m7"],
    "ring": ["no2_last", "no2_age", "no2_m7", "ring_last", "ring_m7"],
}


def no2_table() -> pd.DataFrame:
    """Per origin day D: what is known from the passes of days D-7 .. D-1."""
    s = no2_series()
    s = s[s["product"] == "OFFL"] if (s["product"] == "OFFL").sum() > 1000 else s
    days = pd.date_range(s.index.min(), pd.Timestamp.today().normalize() + pd.Timedelta(days=1), freq="D")
    city, ring = s["no2"].reindex(days), s["no2_ring"].reindex(days)
    out = pd.DataFrame(index=days)
    for name, v in (("no2", city), ("ring", ring)):
        prev = v.shift(1)  # the origin day sees only passes before it
        last_day = pd.Series(np.where(prev.notna(), np.arange(len(days)), np.nan), index=days).ffill()
        age = np.arange(len(days)) - last_day.values + 1  # 1 = yesterday's pass
        last = prev.ffill()
        out[f"{name}_last"] = np.where(age <= WINDOW, last, np.nan)
        out[f"{name}_m7"] = prev.rolling(WINDOW, min_periods=2).mean()
        if name == "no2":
            out["no2_age"] = np.where(age <= WINDOW, age, np.nan)
    return out


def attach(X: pd.DataFrame, origins: pd.DatetimeIndex, table: pd.DataFrame) -> pd.DataFrame:
    t = table.reindex(pd.DatetimeIndex(origins).normalize())
    cols = sorted({c for v in VARIANTS.values() for c in v})
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(t[cols].values, columns=cols)], axis=1)


def split(group, cut, end, wf_fc, cams, table, val_step=3):
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
    o = X.index.get_level_values("origin")[ok]
    X, y, T = X[ok], y[ok], T[ok]
    w = np.asarray(_weights(T, X["src"].values), float)
    X = attach(X, o, table)
    end = min(pd.Timestamp(end) if end else pm.dropna().index.max(), pm.dropna().index.max() - pd.Timedelta(hours=1))
    vo = pd.date_range(cut, end, freq=f"{val_step}h")
    vo = vo[lp_all.ffill(limit=2).reindex(vo).notna().values]
    Xv = hourly_rows(vo, np.arange(lo, hi + 1), lp_all, pm_state(lp_all), same_hour_mean(lp_all), clim, wf_fc, None, cams=cams)
    Xv["src"] = 1.0
    Tv = target_times(Xv)
    act = aqi.pm_to_aqi(pm.reindex(Tv).values)
    okv = ~np.isnan(act) & ~np.isnan(Xv["lag0"].values)
    meta = pd.DataFrame({"origin": Xv.index.get_level_values("origin")[okv], "h": Xv.index.get_level_values("h")[okv],
                         "actual": act[okv]})
    Xv = attach(Xv[okv], meta["origin"], table)
    return X, y, w, Xv, meta


def run() -> pd.DataFrame:
    wx = load_weather()
    table = no2_table()
    wf_fc = splice_forecast_weather(wx, load_prev_runs_ensemble())
    cams_all = pd.read_parquet(RAW / "weather" / "cams.parquet")
    no2_cols = sorted({c for v in VARIANTS.values() for c in v})
    res = []
    for y, (cut, end) in SEASONS.items():
        for group in H_GROUPS:
            t0 = time.time()
            vname = CHOICE[f"{group[0]}-{group[1]}"]
            kw = HOURLY_VARIANTS[vname]
            cams = cams_all if kw.get("use_cams") else None
            X, yy, w, Xv, meta = split(group, pd.Timestamp(cut), end, wf_fc, cams, table)
            base = [c for c in X.columns if X[c].notna().any() and c not in no2_cols]
            out = meta.assign(season=y, group=f"{group[0]}-{group[1]}")
            for name, cols in {"current": base, **{k: base + v for k, v in VARIANTS.items()}}.items():
                preds = []
                for seed in SEEDS:
                    m = _gbm(HOURLY_PARAMS, kw.get("rounds", 300), seed, X[cols], yy, w, "lgbm", kw.get("lgbm_extra"))
                    preds.append(m.predict(Xv[cols].values) + Xv["lag0"].values)
                out[name] = aqi.pm_to_aqi(np.clip(np.expm1(np.mean(preds, axis=0)), 0, None))
                out[f"{name}_s"] = np.std([np.abs(aqi.pm_to_aqi(np.clip(np.expm1(p), 0, None)) - meta["actual"]).mean()
                                           for p in preds])
            res.append(out)
            maes = {k: np.abs(out[k] - out["actual"]).mean() for k in ("current", *VARIANTS)}
            print(f"{y} {group}: " + "  ".join(f"{k} {v:.2f}" for k, v in maes.items())
                  + f"  ({len(out)} rows, {time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(PROC / "no2_test.parquet")
    return df


def boot(d: pd.DataFrame, a: str, b: str, n: int = 2000) -> tuple[float, float, float]:
    """MAE(b) - MAE(a) and its 95% interval, resampling days of the forecast origin."""
    e = (np.abs(d[b] - d["actual"]) - np.abs(d[a] - d["actual"])).groupby(d["origin"].dt.normalize())
    s, c = e.sum().values, e.count().values
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarise(df: pd.DataFrame) -> dict:
    bands = {"1-24": (1, 24), "25-72": (25, 72), "73-168": (73, 168), "1-168": (1, 168)}
    tab = {}
    for y in SEASONS:
        d = df[df["season"] == y]
        tab[y] = {k: {b: round(float(np.abs(d.loc[d["h"].between(*r), k] - d.loc[d["h"].between(*r), "actual"]).mean()), 2)
                      for b, r in bands.items()} for k in ("current", *VARIANTS)}
    sel = df[df["season"] == 2024]
    mae24 = {k: np.abs(sel[k] - sel["actual"]).mean() for k in VARIANTS}
    chosen = min(mae24, key=mae24.get)
    cur24 = np.abs(sel["current"] - sel["actual"]).mean()
    groups = [g for g, d in sel.groupby("group")
              if np.abs(d[chosen] - d["actual"]).mean() < np.abs(d["current"] - d["actual"]).mean()]
    df = df.copy()
    df["router"] = np.where(df["group"].isin(groups), df[chosen], df["current"])
    verdict = {}
    for y in SEASONS:
        d = df[df["season"] == y]
        verdict[y] = {k: {b: [round(x, 2) for x in boot(d[d["h"].between(*r)], "current", k)] for b, r in bands.items()}
                      for k in (chosen, "router")}
    seed_sd = {y: round(float(df[df["season"] == y][[f"{k}_s" for k in ("current", *VARIANTS)]].mean().mean()), 2) for y in SEASONS}
    out = {"mae": tab, "chosen_on_2024": chosen, "chosen_beats_current_2024": bool(mae24[chosen] < cur24),
           "router_groups": groups, "diff_vs_current": verdict, "seed_sd": seed_sd,
           "rows": {y: int((df["season"] == y).sum()) for y in SEASONS}}
    print(json.dumps(out, indent=1))
    (PROC / "no2_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    df = pd.read_parquet(PROC / "no2_test.parquet") if "--summary" in sys.argv else run()
    summarise(df)
