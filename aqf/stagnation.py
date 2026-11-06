"""Does the model misjudge stagnant air, or is the pollution level itself off? (past winters)

For every day of the winters 2020/21 to 2026/27 the forecast for that day is re-made as of
06:00 the day before, with the weather that actually happened (so weather-forecast errors are
out), by the production per-horizon models trained only on data before 1 October of that
winter (a fixed model per winter; production retrains weekly, so its level drifts less). The
forecast's stagnation part (SHAP, AQI points, as on the site's week tab: wind near the ground
and aloft, calm, inversions, pressure) is set against its error:

  if the model overrates (or underrates) stagnant air, the error grows with that part;
  if the pollution level itself is off (sources, transport), the error does not depend on it.

The error is fitted as  error = a + b * stagnation  per winter and over all of them; b * S is
the stagnation-related part of a day's error, the rest is level plus noise.

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
        .venv/bin/python -m aqf.stagnation      # -> data/processed/stagnation.parquet + summary
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import aqi
from .backtest import season_bounds
from .data import PROC, RAW, load_prev_runs_ensemble, load_target, load_weather, splice_forecast_weather
from .explain import GROUPS, group_of
from .features import hourly_rows, pm_state, same_hour_mean, weather_features
from .models import daily_aqi, make_hourly_model

DAY_GROUPS = ["13-18", "19-24", "25-36", "37-48"]  # hours 18-41 from 06:00 = the next day
OUT = PROC / "stagnation.parquet"


def day_parts(router, origins: pd.DatetimeIndex, pm: pd.Series, wfeat) -> pd.DataFrame:
    """Per origin: the next day's forecast and its parts in AQI points (as aqf.explain)."""
    lp = np.log1p(pm)
    state, sh7 = pm_state(lp), same_hour_mean(lp)
    frames = []
    for (lo, hi), name in router.choice.items():
        v = router.variants[name]
        X = hourly_rows(origins, np.arange(lo, hi + 1), lp, state, sh7, v.clim, wfeat, None,
                        cams=getattr(v, "cams", None), lead_matched_vent=True)
        X["src"] = 1.0
        h = X.index.get_level_values("h").values
        for (blo, bhi), booster in v.boosters.items():
            sel = (h >= blo) & (h <= bhi)
            if not sel.any():
                continue
            cols = v.features if isinstance(v.features, list) else v.features[(blo, bhi)]
            C = booster.predict(X.loc[sel, cols].values, pred_contrib=True)
            g = pd.DataFrame(C[:, :-1], columns=cols).T.groupby([group_of(c) for c in cols]).sum().T
            g["other"] = g.get("other", 0.0) + C[:, -1]
            g["start"] = X.loc[sel, v.base].values
            g["origin"] = X.index.get_level_values("origin")[sel]
            g["h"] = h[sel]
            frames.append(g)
    G = pd.concat(frames, ignore_index=True).fillna(0.0)
    cum = G["start"].values.copy()
    prev = aqi.pm_to_aqi(np.expm1(cum))
    steps = {}
    for grp in GROUPS:
        cum = cum + (G[grp].values if grp in G else 0.0)
        cur = aqi.pm_to_aqi(np.clip(np.expm1(cum), 0, None))
        steps[grp], prev = cur - prev, cur
    T = pd.to_datetime(G["origin"]) + pd.to_timedelta(G["h"], unit="h")
    df = pd.DataFrame({"origin": G["origin"], "date": T.dt.normalize(), "forecast": prev, **steps})
    df = df[df["date"] == pd.to_datetime(df["origin"]).dt.normalize() + pd.Timedelta(days=1)]
    agg = df.groupby("date").agg(n=("forecast", "size"), forecast=("forecast", "mean"), **{g: (g, "mean") for g in GROUPS})
    return agg[agg["n"] == 24].drop(columns="n")


def run(seasons=range(2020, 2027)) -> pd.DataFrame:
    wx = load_weather()
    wf_act = weather_features(wx)
    wf_fc = splice_forecast_weather(wx, load_prev_runs_ensemble())  # training, as in production
    choice_all = json.loads((PROC / "horizon_choice.json").read_text())
    choice = {k: choice_all[k] for k in DAY_GROUPS}
    cams = pd.read_parquet(RAW / "weather" / "cams.parquet")
    out = []
    for y in seasons:
        t0 = time.time()
        cut, end = season_bounds(y)
        tgt, _ = load_target(profile_cutoff=cut)
        pm = tgt["pm"]
        last = min(end, pm.dropna().index.max() - pd.Timedelta(days=2))
        origins = pd.date_range(cut + pd.Timedelta(hours=6), last, freq="D")
        origins = origins[np.log1p(pm).ffill(limit=2).reindex(origins).notna().values]
        if not len(origins):
            continue
        router = make_hourly_model(choice, cams).fit(pm, tgt["src"], wf_fc, cutoff=cut)
        d = day_parts(router, origins, pm, wf_act)
        d["actual"] = daily_aqi(pm).reindex(d.index)
        d["season"] = y
        out.append(d.dropna(subset=["actual"]))
        print(f"winter {y}/{y + 1}: {len(out[-1])} days, {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(out)
    df["error"] = df["forecast"] - df["actual"]
    df.to_parquet(OUT)
    return df


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with 95% half-widths of the coefficients."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / max(1, len(y) - X.shape[1])
    return beta, 1.96 * np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))


def summary(df: pd.DataFrame) -> dict:
    def fit(g):
        b, a = np.polyfit(g["stagnation"], g["error"], 1)
        r = np.corrcoef(g["stagnation"], g["error"])[0, 1]
        out = {"days": len(g), "mean_error": round(float(g["error"].mean()), 1), "mae": round(float(g["error"].abs().mean()), 1),
               "mean_stagnation": round(float(g["stagnation"].mean()), 1), "a": round(float(a), 1), "b": round(float(b), 2),
               "r": round(float(r), 2), "r2": round(float(r * r), 3)}
        # the same slope with a separate level for every month: a level that shifts within the
        # winter can no longer pass for a stagnation effect
        months = g.index.to_period("M").astype(str)
        if months.nunique() > 1:
            M = pd.get_dummies(months).values.astype(float)
            beta, hw = _ols(np.column_stack([g["stagnation"].values, M]), g["error"].values)
            out.update(b_month=round(float(beta[0]), 2), b_month_ci=round(float(hw[0]), 2))
        return out
    res = {"all": fit(df), "by_winter": {str(y): fit(g) for y, g in df.groupby("season")},
           "from": f"{df.index.min():%Y-%m-%d}", "to": f"{df.index.max():%Y-%m-%d}"}
    # the current winter: few days, so also with a time trend (stagnation often builds up over weeks)
    cur = df[df["season"] == df["season"].max()]
    if len(cur) >= 10:
        t = (cur.index - cur.index.min()).days.values.astype(float)
        one = np.ones(len(cur))
        b1, h1 = _ols(np.column_stack([one, cur["stagnation"].values]), cur["error"].values)
        b2, h2 = _ols(np.column_stack([one, cur["stagnation"].values, t]), cur["error"].values)
        res["current"] = {"season": int(cur["season"].iloc[0]), "days": int(len(cur)),
                          "b": round(float(b1[1]), 2), "b_ci": round(float(h1[1]), 2),
                          "b_time": round(float(b2[1]), 2), "b_time_ci": round(float(h2[1]), 2),
                          "corr_time": round(float(np.corrcoef(cur["stagnation"], t)[0, 1]), 2)}
    q = pd.qcut(df["stagnation"], 5)
    res["by_stagnation_quintile"] = [{"stagnation": f"{iv.left:.0f}..{iv.right:.0f}", "days": int(len(g)),
                                      "mean_error": round(float(g["error"].mean()), 1), "mae": round(float(g["error"].abs().mean()), 1)}
                                     for iv, g in df.groupby(q, observed=True)]
    b = res["all"]["b"]
    m = df.assign(stag_part=b * df["stagnation"]).groupby(df.index.to_period("M"))
    res["by_month"] = [{"month": str(k), "days": int(len(g)), "mean_error": round(float(g["error"].mean()), 1),
                        "stagnation_part": round(float(g["stag_part"].mean()), 1),
                        "level_part": round(float((g["error"] - g["stag_part"]).mean()), 1)} for k, g in m]
    (PROC / "stagnation.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    print(json.dumps(summary(run()), indent=1, ensure_ascii=False))
