"""Leakage test: does anything after the forecast origin influence training or the forecast?

For several cutoffs (= backtest training cutoff = forecast origin) the raw AQI data after
the origin is replaced by random noise. If the pipeline is leak-free, the training targets,
the fitted models and the forecasts issued at the origin are bit-for-bit identical.

A negative control re-runs the check with the old centred smoothing patched in, to show
the test does catch a one-hour look-ahead.

Weather runs (since 2026-11-05): with archived forecasts by lead day, the forecast issued at the
origin may only use frame L up to origin + 24·L h (later values come from runs issued after the
origin). Those values are replaced by noise; the forecast must not change. Its negative control
is the lead-1 ventilation the scoring used until 2026-11-05 (research/forecast_trust_test.py).

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
        .venv/bin/python -m aqf.leakage_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, models
from .data import EMBASSY, IQAIR_JSON, load_prev_runs_ensemble, load_target, load_weather, splice_forecast_weather
from .models import HourlyModel, LongRange
from .sources import airnow, tashkentairdata

CUTOFFS = ["2023-12-01", "2024-11-01", "2025-02-01", "2025-12-01", "2026-10-14"]
DETERMINISTIC = dict(deterministic=True, force_row_wise=True)


def scramble(s: pd.Series, origin: pd.Timestamp, lo: float, hi: float, seed: int) -> pd.Series:
    s = s.copy()
    m = s.index > origin
    s[m] = np.random.default_rng(seed).uniform(lo, hi, m.sum())
    return s


def after_issue(wf_fc: dict, origin: pd.Timestamp, seed: int) -> dict:
    """wf_fc with every value of lead frame L after origin + 24·L h replaced by noise."""
    rng = np.random.default_rng(seed)
    out = {}
    for L, f in wf_fc.items():
        f = f.copy()
        m = f.index > origin + pd.Timedelta(hours=24 * L)
        f.loc[m] = rng.normal(f.mean().values, f.std().values + 1e-9, size=(int(m.sum()), f.shape[1]))
        out[L] = f
    return out


def weather_diff(hm, origin: pd.Timestamp, pm: pd.Series, wf_fc: dict) -> float:
    a = hm.predict_frame(pd.DatetimeIndex([origin]), pm, wf_fc)["pred_aqi"].values
    b = hm.predict_frame(pd.DatetimeIndex([origin]), pm, after_issue(wf_fc, origin, 3))["pred_aqi"].values
    return float(np.nanmax(np.abs(a - b)))


def run_one(origin: pd.Timestamp, iq_aqi: pd.Series, emb: pd.Series, wf_fc) -> dict:
    tgt, cal = load_target(profile_cutoff=origin, iq_aqi=iq_aqi, emb_raw=emb)
    pm, src = tgt["pm"], tgt["src"]
    hm = HourlyModel(max_h=168, lgbm_extra=DETERMINISTIC).fit(pm, src, wf_fc, cutoff=origin)
    hp = hm.predict_frame(pd.DatetimeIndex([origin]), pm, wf_fc)["pred_aqi"].values
    lr = LongRange().fit(pm, cutoff=origin)
    day = origin.normalize() - pd.Timedelta(days=1)
    lp = lr.predict_frame(pd.DatetimeIndex([day]), pm)["pred_aqi"].values
    return {"past_target": pm[pm.index <= origin], "cal": (cal.a, cal.b), "hourly": hp, "daily": lp,
            "model": hm, "pm": pm}


def compare(origin: pd.Timestamp, wf_fc) -> dict:
    iq = tashkentairdata.load(IQAIR_JSON)
    emb = airnow.load_hourly(EMBASSY)
    a = run_one(origin, iq, emb, wf_fc)
    b = run_one(origin, scramble(iq, origin, 0, 500, 1), scramble(emb, origin, 0, 400, 2), wf_fc)
    t1, t2 = a["past_target"], b["past_target"]
    return {
        "origin": str(origin),
        "target_up_to_origin_identical": bool(np.allclose(t1.values, t2.values, equal_nan=True)),
        "calibration_identical": a["cal"] == b["cal"],
        "hourly_forecast_max_diff": float(np.nanmax(np.abs(a["hourly"] - b["hourly"]))),
        "daily_forecast_max_diff": float(np.nanmax(np.abs(a["daily"] - b["daily"]))),
        "weather_after_issue_max_diff": weather_diff(a["model"], origin, a["pm"], wf_fc),
        "_model": a["model"], "_pm": a["pm"],
    }


def main() -> None:
    wf_fc = splice_forecast_weather(load_weather(), load_prev_runs_ensemble())
    rows = [compare(pd.Timestamp(c), wf_fc) for c in CUTOFFS]
    print(pd.DataFrame(rows).drop(columns=["_model", "_pm"]).to_string(index=False))

    # negative control for the weather check: the lead-1 ventilation used until 2026-11-05
    r = next(x for x in rows if x["origin"] == str(pd.Timestamp("2024-11-01")))
    keep_rows = models.hourly_rows
    models.hourly_rows = lambda *a, **k: keep_rows(*a, **{**k, "lead_matched_vent": False})
    try:
        d = weather_diff(r["_model"], pd.Timestamp(r["origin"]), r["_pm"], wf_fc)
    finally:
        models.hourly_rows = keep_rows
    print(f"\nnegative control (lead-1 ventilation, origin {r['origin']}): weather_after_issue_max_diff {d:.3f}")

    # negative control: centred smoothing peeks one hour ahead -> must be detected
    keep = data.smooth_spikes
    data.smooth_spikes = lambda lp, a: (0.25 * lp.shift(1).fillna(lp) + 0.5 * lp
                                        + 0.25 * lp.shift(-1).fillna(lp)).where(lp.notna())
    try:
        ctl = compare(pd.Timestamp("2024-11-01"), wf_fc)
    finally:
        data.smooth_spikes = keep
    print("\nnegative control (deliberate 1-hour look-ahead):")
    print(pd.DataFrame([ctl]).drop(columns=["_model", "_pm"]).to_string(index=False))


if __name__ == "__main__":
    main()
