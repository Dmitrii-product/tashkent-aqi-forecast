"""Tashkent's central heating season, and a test of it as an input to the hourly model.

Dates: official starts and ends where the press reported them (circulation start, i.e.
when the boilers fire up). The city cites a rule (start the day after five consecutive days
with a daily mean below +8 °C), but in practice it starts 1–5 weeks earlier: where no start
was reported, the rule's date is moved by the median gap of the reported years; where no
end was reported, 15 March (reported ends: 10, 18 and 24 March). Daily means come from the
weather archive.

The model would see only what is known at the forecast time: whether heating is on at the
origin and for how many days it has been on (no look at future decisions).

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
        .venv/bin/python -m aqf.heating          # head-to-head test -> data/processed/heating_test.json
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import aqi
from .data import PROC, RAW, load_prev_runs_ensemble, load_target, load_weather, splice_forecast_weather
from .features import Climatology, hourly_rows, pm_state, same_hour_mean, target_times
from .models import HOURLY_PARAMS, HOURLY_VARIANTS, _gbm, _weights

# season (year it starts) -> (start, end); None = not reported, the rule fills it in
OFFICIAL = {
    2019: ("2019-10-25", None),          # kun.uz / norma.uz 2019
    2021: ("2021-10-12", None),          # gazeta.uz 2024-10-31 (social objects 12th, homes 14th)
    2022: ("2022-10-22", "2023-03-10"),  # gazeta.uz; kun.uz 2023-03-06
    2023: ("2023-11-03", None),          # afisha.uz 2023-11-02 (circulation 3rd, homes 6th)
    2024: ("2024-11-05", "2025-03-18"),  # kun.uz 2024-11-04; gazeta.uz / kun.uz 2025-03-18
    2025: ("2025-11-03", "2026-03-24"),  # gazeta.uz 2025-11-01; gazeta.uz 2026-03-19
}
THRESHOLD_C, RUN_DAYS = 8.0, 5
AGE_CAP = 45  # days since the start, capped


def rule_dates(t_daily: pd.Series) -> dict[int, tuple[pd.Timestamp, pd.Timestamp | None]]:
    """Start / end of each season by the +8 °C five-day rule."""
    cold = (t_daily < THRESHOLD_C).astype(int)
    warm = (t_daily > THRESHOLD_C).astype(int)
    run_cold = cold.rolling(RUN_DAYS).sum() == RUN_DAYS
    run_warm = warm.rolling(RUN_DAYS).sum() == RUN_DAYS
    out = {}
    for y in range(t_daily.index.min().year, t_daily.index.max().year + 1):
        a = run_cold[(run_cold.index >= f"{y}-09-15") & (run_cold.index < f"{y + 1}-01-31")]
        b = run_warm[(run_warm.index >= f"{y + 1}-02-15") & (run_warm.index < f"{y + 1}-05-31")]
        start = a.index[a.values][0] + pd.Timedelta(days=1) if a.any() else None
        end = b.index[b.values][0] + pd.Timedelta(days=1) if b.any() else None
        if start is not None:
            out[y] = (start, end)
    return out


def seasons(t_daily: pd.Series) -> pd.DataFrame:
    """Start / end per season: official where reported; otherwise the shifted rule (start)
    and 15 March (end)."""
    rule = rule_dates(t_daily)
    gaps = [(pd.Timestamp(OFFICIAL[y][0]) - rule[y][0]).days for y in OFFICIAL if OFFICIAL[y][0] and y in rule]
    shift = pd.Timedelta(days=round(float(np.median(gaps)))) if gaps else pd.Timedelta(0)
    rows = []
    for y in sorted(set(rule) | set(OFFICIAL)):
        rs, re_ = rule.get(y, (None, None))
        os_, oe = OFFICIAL.get(y, (None, None))
        rows.append({"season": y, "start": pd.Timestamp(os_) if os_ else rs + shift,
                     "end": pd.Timestamp(oe) if oe else pd.Timestamp(f"{y + 1}-03-15"),
                     "start_src": "official" if os_ else f"rule {shift.days:+d} d", "end_src": "official" if oe else "15 Mar",
                     "rule_start": rs, "rule_end": re_})
    return pd.DataFrame(rows)


def features(times: pd.DatetimeIndex, table: pd.DataFrame) -> pd.DataFrame:
    """Per time: heating on (0/1) and days since this season's start (0 when off)."""
    d = pd.DatetimeIndex(times).normalize()
    on, age = np.zeros(len(d)), np.zeros(len(d))
    for r in table.itertuples():
        if r.start is None or pd.isna(r.start):
            continue
        end = r.end if r.end is not None and not pd.isna(r.end) else pd.Timestamp.max.normalize()
        m = (d >= r.start) & (d < end)
        on[m] = 1.0
        age[m] = np.minimum((d[m] - r.start).days, AGE_CAP)
    return pd.DataFrame({"heat_o": on, "heat_age_o": age})


# ---------------------------------------------------------------- head-to-head test
GROUPS = {(13, 18): "pred_v3_mm_cams", (37, 48): "tuned_37_48", (97, 120): "pred_v3_mm_l1", (145, 168): "pred_v3_mm_big"}
SEASONS = {2024: ("2024-10-01", "2025-04-01"), 2025: ("2025-10-01", "2026-04-01"), 2026: ("2026-10-01", None)}


def _split(group, cut, end, wf_fc, cams, table, seed=0, val_step=3):
    """Training rows before `cut`, validation forecasts every `val_step` h in [cut, end),
    as in aqf.tune, with the heating features attached by origin time."""
    lo, hi = group
    tgt, _ = load_target(profile_cutoff=cut)
    pm, src = tgt["pm"], tgt["src"]
    lp_all = np.log1p(pm)
    lp = lp_all[lp_all.index < cut]
    clim = Climatology().fit(lp)
    state, sh7 = pm_state(lp), same_hour_mean(lp)
    rng = np.random.default_rng(seed)
    origins = state.index[(state.index.hour % 2 == 0) & state["lag0"].notna()]
    k = min(4, hi - lo + 1)
    H = rng.integers(lo, hi + 1, size=(len(origins), k)) if hi > lo else np.full((len(origins), 1), lo)
    X = hourly_rows(origins, H, lp, state, sh7, clim, wf_fc, src, loo=True, cams=cams)
    T = target_times(X)
    y = lp.reindex(T).values - X["lag0"].values
    ok = ~np.isnan(y)
    X, y, T = X[ok], y[ok], T[ok]
    w = np.asarray(_weights(T, X["src"].values), float)
    Xh = features(X.index.get_level_values("origin"), table)
    X = pd.concat([X.reset_index(drop=True), Xh], axis=1)
    end = min(pd.Timestamp(end) if end else pm.dropna().index.max(), pm.dropna().index.max() - pd.Timedelta(hours=1))
    vo = pd.date_range(cut, end, freq=f"{val_step}h")
    vo = vo[lp_all.ffill(limit=2).reindex(vo).notna().values]
    Xv = hourly_rows(vo, np.arange(lo, hi + 1), lp_all, pm_state(lp_all), same_hour_mean(lp_all), clim, wf_fc, None, cams=cams)
    Xv["src"] = 1.0
    Tv = target_times(Xv)
    act = aqi.pm_to_aqi(pm.reindex(Tv).values)
    okv = ~np.isnan(act) & ~np.isnan(Xv["lag0"].values)
    Xvh = features(Xv.index.get_level_values("origin")[okv], table)
    Xv = pd.concat([Xv[okv].reset_index(drop=True), Xvh], axis=1)
    return X, y, w, Xv, act[okv]


def test() -> dict:
    wx = load_weather()
    table = seasons(wx["temperature_2m"].resample("D").mean())
    print(table.to_string(index=False), flush=True)
    wf_fc = splice_forecast_weather(wx, load_prev_runs_ensemble())
    cams_all = pd.read_parquet(RAW / "weather" / "cams.parquet")
    res = []
    for y, (cut, end) in SEASONS.items():
        for group, variant in GROUPS.items():
            t0 = time.time()
            kw = HOURLY_VARIANTS[variant]
            cams = cams_all if kw.get("use_cams") else None
            X, yy, w, Xv, act = _split(group, cut, end, wf_fc, cams, table)
            base_cols = [c for c in X.columns if X[c].notna().any() and not c.startswith("heat_")]
            for name, cols in (("current", base_cols), ("heating", base_cols + ["heat_o", "heat_age_o"])):
                maes = []
                for seed in (0, 1):
                    m = _gbm(HOURLY_PARAMS, kw.get("rounds", 300), seed, X[cols], yy, w, "lgbm", kw.get("lgbm_extra"))
                    pred = aqi.pm_to_aqi(np.expm1(m.predict(Xv[cols].values) + Xv["lag0"].values))
                    maes.append(float(np.nanmean(np.abs(pred - act))))
                res.append({"season": y, "group": f"{group[0]}-{group[1]}", "model": name,
                            "mae": float(np.mean(maes)), "n": int(len(act))})
            cur, heat = res[-2]["mae"], res[-1]["mae"]
            print(f"season {y} group {group}: current {cur:.2f}  heating {heat:.2f}  diff {heat - cur:+.2f}  "
                  f"({len(act)} rows, {time.time() - t0:.0f}s)", flush=True)
    out = {"table": json.loads(table.to_json(orient="records", date_format="iso")), "results": res}
    (PROC / "heating_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    test()
